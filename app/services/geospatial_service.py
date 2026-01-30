
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class GeospatialService:
    def __init__(self):
        # UserAgent is required by Nominatim
        self.geolocator = Nominatim(user_agent="sahayog_setu_v1")

    def get_coordinates(self, place_name: str) -> Optional[Tuple[float, float]]:
        """
        Fetch lat, lon for a given place name (e.g. "Kuduragere, Karnataka")
        """
        try:
            location = self.geolocator.geocode(place_name)
            if location:
                return (location.latitude, location.longitude)
            return None
        except Exception as e:
            logger.error(f"Geocoding error for {place_name}: {e}")
            return None

    def calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """
        Calculate geodesic distance (in km) between two (lat, lon) tuples.
        """
        try:
            # geodesic((lat1, lon1), (lat2, lon2)).km
            return geodesic(loc1, loc2).km
        except Exception as e:
            logger.error(f"Distance calculation error: {e}")
            return float('inf') # Return infinite distance on error
