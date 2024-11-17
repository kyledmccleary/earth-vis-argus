import rasterio
import numpy as np
import os
from scipy.spatial import transform


class GeoTIFFCache:
    def __init__(self, geotiff_folder):
        self.geotiff_folder = geotiff_folder
        self.cache = {}

    def load_geotiff_data(self, region):
        if region in self.cache:
            return self.cache[region]
        
        region_folder = os.path.join(self.geotiff_folder, region)
        if not os.path.exists(region_folder):
            self.cache[region] = (None, None)
            return self.cache[region]
        region_files = os.listdir(region_folder)
        if not region_files:
            self.cache[region] = (None, None)
            return self.cache[region]
        
        selected_file = np.random.choice(region_files)
        file_path = os.path.join(region_folder, selected_file)
        with rasterio.open(file_path) as src:
            data = src.read()
            data = np.moveaxis(data, 0, -1)
            trans = src.transform
        self.cache[region] = (data, trans)
        return self.cache[region]

    def clear_cache(self):
        self.cache = {}



# define camera
class Camera:
    def __init__(self, resolution, fov, position, orientation):
        """
        Initialize the camera parameters.

        Parameters:
            resolution (tuple): Resolution of the camera (width, height).
            fov (float): Field of view in degrees (assumes square FOV).
            position (np.ndarray): Camera position in ECEF (3,).
            orientation (np.ndarray): 3x3 rotation matrix for orientation.
        """
        self.resolution = resolution
        self.fov = np.radians(fov)  # Convert FOV to radians
        self.position = np.array(position)
        self.orientation = np.array(orientation)

    def ray_directions(self):
        """
        Generate ray directions for the camera.

        Returns:
            np.ndarray: Array of ray directions (HxWx3) in the camera frame.
        """
        width, height = self.resolution
        half_width = np.tan(self.fov / 2)
        half_height = half_width * (height / width)

        x = np.linspace(-half_width, half_width, width)
        y = np.linspace(-half_height, half_height, height)
        xx, yy = np.meshgrid(x, y)
        zz = np.ones_like(xx)  # Assume unit depth

        # Stack and normalize ray directions
        ray_directions = np.stack([xx, yy, zz], axis=-1)
        ray_directions /= np.linalg.norm(ray_directions, axis=-1, keepdims=True)
        return ray_directions

    def rays_in_ecef(self):
        """
        Transform ray directions from the camera frame to the ECEF frame.

        Returns:
            np.ndarray: Array of ray directions (HxWx3) in the ECEF frame.
        """
        return self.ray_directions() @ self.orientation.T

def get_nadir_rotation(satellite_position):
    # pointing nadir in world coordinates
    x, y, z = satellite_position
    zc = dir_vector = -np.array([x, y, z]) / np.linalg.norm([x, y, z])
    axis_of_rotation_z = np.cross(np.array([0,0,1]), dir_vector)
    rc = axis_of_rotation_z = axis_of_rotation_z / np.linalg.norm(axis_of_rotation_z)
    xc = -rc 

    yc = south_vector = np.cross(rc, zc)
    R = np.stack([xc, yc, zc], axis=-1)
    return R


def intersect_ellipsoid(ray_directions, satellite_position, a = 6378137.0, b = 6356752.314245):
    """
    Vectorized computation of ray intersections with the WGS84 ellipsoid.

    Parameters:
        ray_directions (np.ndarray): Array of ray directions (Nx3).
        satellite_position (np.ndarray): Satellite position in ECEF (3,).
        a (float): Semi-major axis of the WGS84 ellipsoid (meters).
        b (float): Semi-minor axis of the WGS84 ellipsoid (meters).

    Returns:
        np.ndarray: Intersection points (Nx3), or NaN for rays that miss.
    """
    H, W, _ = ray_directions.shape
    ray_directions_flat = ray_directions.reshape(-1, 3)


    A = ray_directions_flat[:, 0]**2 / a**2 + ray_directions_flat[:, 1]**2 / a**2 + ray_directions_flat[:, 2]**2 / b**2
    B = 2 * (satellite_position[0] * ray_directions_flat[:, 0] / a**2 + 
                satellite_position[1] * ray_directions_flat[:, 1] / a**2 + 
                satellite_position[2] * ray_directions_flat[:, 2] / b**2)
    C = (satellite_position[0]**2 / a**2 +
            satellite_position[1]**2 / a**2 +
            satellite_position[2]**2 / b**2 - 1)
    discriminant = B**2 - 4 * A * C

    # Initialize intersection points as NaN
    intersection_points_flat = np.full_like(ray_directions_flat, np.nan)

    valid_mask = discriminant >= 0
    if np.any(valid_mask):
        # Compute roots of the quadratic equation
        sqrt_discriminant = np.sqrt(discriminant[valid_mask])
        t1 = (-B[valid_mask] - sqrt_discriminant) / (2 * A[valid_mask])
        t2 = (-B[valid_mask] + sqrt_discriminant) / (2 * A[valid_mask])

        # Choose the smallest positive t
        t = np.where((t1 > 0) & ((t1 < t2) | (t2 <= 0)), t1, t2)
        t = np.where(t > 0, t, np.nan)  # Filter out negative t values

        # Calculate intersection points
        valid_ray_directions = ray_directions_flat[valid_mask]
        intersection_points_flat[valid_mask] = t[:, None] * valid_ray_directions + satellite_position
    # Reshape intersection points back to original ray grid shape
    intersection_points = intersection_points_flat.reshape(H, W, 3)
    return intersection_points


def convert_to_lat_lon(intersection_points, a = 6378137.0, b = 6356752.314245):
    """
    Convert intersection points (ECEF) to latitude and longitude.

    Parameters:
        intersection_points (np.ndarray): Array of intersection points (HxWx3) in ECEF coordinates.

    Returns:
        np.ndarray: Array of latitude and longitude (HxWx2), or NaN for invalid points.
    """

    H, W, _ = intersection_points.shape
    intersection_points_flat = intersection_points.reshape(-1, 3)

    valid_mask = ~np.isnan(intersection_points_flat).any(axis=1)
    
    lat_lon_flat = np.full((H * W, 2), np.nan)

    valid_points = intersection_points_flat[valid_mask]

    x, y, z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]

    # Longitude calculation (same for geodetic and geocentric)
    lon = np.degrees(np.arctan2(y, x))

    # Geodetic latitude calculation (iterative approach)
    e2 = (a**2 - b**2) / a**2  # First eccentricity squared
    ep2 = (a**2 - b**2) / b**2  # Second eccentricity squared
    p = np.sqrt(x**2 + y**2)

    # Initial approximation of latitude
    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(z + ep2 * b * np.sin(theta)**3, p - e2 * a * np.cos(theta)**3)

    # Convert to degrees
    lat = np.degrees(lat)

    # Store results in flat array
    lat_lon_flat[valid_mask, 0] = lat
    lat_lon_flat[valid_mask, 1] = lon

    return lat_lon_flat.reshape(H, W, 2)


hfov = 66.1
width = 640
height = 360
resolution = np.array([width, height])
satellite_position = np.array([983017.6742974258, -6109867.766065873, 3098940.646932125])
orientation = get_nadir_rotation(satellite_position)
camera = Camera(resolution, hfov, satellite_position, orientation)
ray_directions_ecef = camera.rays_in_ecef()
intersection_points = intersect_ellipsoid(ray_directions_ecef, satellite_position)
lat_lon = convert_to_lat_lon(intersection_points)

# Define MGRS latitude bands and UTM exceptions
latitude_bands = [
    {"name": "C", "min_lat": -80, "max_lat": -72},
    {"name": "D", "min_lat": -72, "max_lat": -64},
    {"name": "E", "min_lat": -64, "max_lat": -56},
    {"name": "F", "min_lat": -56, "max_lat": -48},
    {"name": "G", "min_lat": -48, "max_lat": -40},
    {"name": "H", "min_lat": -40, "max_lat": -32},
    {"name": "J", "min_lat": -32, "max_lat": -24},
    {"name": "K", "min_lat": -24, "max_lat": -16},
    {"name": "L", "min_lat": -16, "max_lat": -8},
    {"name": "M", "min_lat": -8, "max_lat": 0},
    {"name": "N", "min_lat": 0, "max_lat": 8},
    {"name": "P", "min_lat": 8, "max_lat": 16},
    {"name": "Q", "min_lat": 16, "max_lat": 24},
    {"name": "R", "min_lat": 24, "max_lat": 32},
    {"name": "S", "min_lat": 32, "max_lat": 40},
    {"name": "T", "min_lat": 40, "max_lat": 48},
    {"name": "U", "min_lat": 48, "max_lat": 56},
    {"name": "V", "min_lat": 56, "max_lat": 64},
    {"name": "W", "min_lat": 64, "max_lat": 72},
    {"name": "X", "min_lat": 72, "max_lat": 84},  # X spans 12° latitude
]

utm_exceptions = [
    {"zone": 32, "min_lon": 3, "max_lon": 12, "bands": ["V"]},  # Norway
    {"zone": 31, "min_lon": 0, "max_lon": 9, "bands": ["X"]},  # Svalbard
    {"zone": 33, "min_lon": 9, "max_lon": 21, "bands": ["X"]},  # Svalbard
    {"zone": 35, "min_lon": 21, "max_lon": 33, "bands": ["X"]},  # Svalbard
    {"zone": 37, "min_lon": 33, "max_lon": 42, "bands": ["X"]},  # Svalbard
]

# Create lookup tables for vectorized latitude band calculation
latitude_band_names = np.array([band["name"] for band in latitude_bands])
latitude_band_edges = np.array([[band["min_lat"], band["max_lat"]] for band in latitude_bands])

def calculate_mgrs_zones(latitudes, longitudes):
    """
    Vectorized computation of MGRS regions for given latitude and longitude arrays.

    Parameters:
        latitudes (np.ndarray): 1D or 2D array of latitudes in degrees.
        longitudes (np.ndarray): 1D or 2D array of longitudes in degrees.

    Returns:
        np.ndarray: Array of MGRS region identifiers (same shape as input).
    """
    # Flatten lat/lon for processing
    lat_flat = latitudes.ravel()
    lon_flat = longitudes.ravel()

    # Determine latitude bands
    lat_bands = np.full(lat_flat.shape, None, dtype=object)
    for i, (min_lat, max_lat) in enumerate(latitude_band_edges):
        mask = (lat_flat >= min_lat) & (lat_flat < max_lat)
        lat_bands[mask] = latitude_band_names[i]

    # Determine UTM zones (default calculation)
    utm_zones = ((lon_flat + 180) // 6 + 1).astype(int)

    # Apply UTM exceptions
    for exception in utm_exceptions:
        mask = (
            (lon_flat >= exception["min_lon"]) &
            (lon_flat < exception["max_lon"]) &
            np.isin(lat_bands, exception["bands"])
        )
        utm_zones[mask] = exception["zone"]

    # Combine UTM zones and latitude bands
    mgrs_regions = np.array([f"{zone}{band}" if band is not None else None
                             for zone, band in zip(utm_zones, lat_bands)])

    # Reshape to match input lat/lon shape
    return mgrs_regions.reshape(latitudes.shape)

def query_pixel_colors(latitudes, longitudes, image_data, trans):
    latitudes_flat = latitudes.flatten()
    longitudes_flat = longitudes.flatten()

    inverse_transform = ~trans

    cols, rows = inverse_transform * (longitudes_flat, latitudes_flat)

    # Round and convert to integers
    cols = np.floor(cols).astype(int)
    rows = np.floor(rows).astype(int)

    # Get image dimensions
    height, width, _ = image_data.shape

    # Create a mask for valid indices
    valid_mask = (
        (rows >= 0) & (rows < height) &
        (cols >= 0) & (cols < width)
    )

    # Prepare an array for the pixel values
    num_pixels = latitudes_flat.size
    num_bands = image_data.shape[-1]
    pixel_values = np.zeros((num_pixels, num_bands), dtype=image_data.dtype)

    # Only retrieve pixel values for valid indices
    if np.any(valid_mask):
        pixel_values[valid_mask] = image_data[rows[valid_mask], cols[valid_mask], :]

    # Handle invalid indices (e.g., set to NaN)
    # pixel_values[~valid_mask] = np.nan  # Uncomment if you prefer NaN for invalid pixels

    # Reshape the output to match the input shape (H x W x bands)
    output_shape = latitudes.shape + (num_bands,)
    pixel_values = pixel_values.reshape(output_shape)

    return pixel_values

lat_lon_flat = lat_lon.reshape(-1, 2)
latitudes = lat_lon_flat[:, 0]
longitudes = lat_lon_flat[:, 1]
mgrs_regions = calculate_mgrs_zones(latitudes, longitudes)
present_regions = np.unique(mgrs_regions)
print(present_regions)

cache = GeoTIFFCache('region_mosaics')

for region in present_regions:

    data, trans = cache.load_geotiff_data(region)
    if data is not None:
        print(f"Found data for region {region}")
    else:
        print(f"No data found for region {region}")
        continue

    region_mask = (mgrs_regions == region)

    if not np.any(region_mask):
        continue

    latitudes_region = latitudes[region_mask]
    longitudes_region = longitudes[region_mask]

    image_data, trans = cache.load_geotiff_data(region)
    pixel_colors_region = query_pixel_colors(
        latitudes_region, longitudes_region, image_data, trans
    )

    if "pixel_colors_full" not in locals():
        pixel_colors_full = np.zeros((width, height, 3), dtype=np.uint8)

    # Flatten the full image and region mask for consistent indexing
    pixel_colors_full_flat = pixel_colors_full.reshape(-1, 3)
    region_mask_flat = region_mask.flatten()

    # Ensure pixel colors match the number of True values in the mask
    assert pixel_colors_region.shape[0] == np.sum(region_mask_flat), \
        "Mismatch between region mask size and pixel values."

    # Assign region-specific pixel values to the flattened full image
    pixel_colors_full_flat[region_mask_flat] = pixel_colors_region

    # Reshape the flat full image back to its original shape
    pixel_colors_full = pixel_colors_full_flat.reshape(height, width, 3)

import matplotlib.pyplot as plt
plt.imshow(pixel_colors_full)
plt.show()
