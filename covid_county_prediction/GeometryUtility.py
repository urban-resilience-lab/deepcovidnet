import pandas as pd
import geopandas as gpd
import glob

import covid_county_prediction.config.GeometryUtilityConfig as config


class GeometryUtility(object):

    @classmethod
    def get_poi_with_geom(cls):
        dfs = []
        for file in glob.glob(config.core_poi_apr_data_path + "core_poi-part*.csv"):
            dfs.append(pd.read_csv(file, usecols=['safegraph_place_id', 'latitude', 'longitude']))
        dfs = pd.concat(dfs)
        gdfs = gpd.GeoDataFrame(dfs, geometry=gpd.points_from_xy(dfs.longitude, dfs.latitude))
        gdfs = gdfs[gdfs.geometry.type == 'Point']  # Drop NoneType - empty geometries
        gdfs.crs = "epsg:4269"  # Assume North American Datum projection (NAD83) - required for spatial join
        return gdfs

    @classmethod
    def get_fips_with_geom(cls):
        svi = gpd.read_file(config.svi_data_us_county_data_path)
        svi = svi[['FIPS', 'geometry']]
        svi = svi[svi.geometry.type == 'Polygon']  # Drop NoneType - empty geometries
        return svi

    @classmethod
    def get_poi_labeled_with_fips(cls):
        """
        Get points of interest identifiers labeled with county FIPS code
        :return: point of information safegraph ID with county FIPS code
        """
        poi_with_geom = cls.get_poi_with_geom()
        fips_with_geom = cls.get_fips_with_geom()
        poi_with_fips = gpd.sjoin(poi_with_geom, fips_with_geom, how='left', op='intersects')
        return poi_with_fips.drop('geometry', axis=1)[['safegraph_place_id', 'FIPS']]


if __name__ == '__main__':
    geometryUtility = GeometryUtility()
    result = geometryUtility.get_poi_labeled_with_fips()
    result.to_csv(config.core_poi_apr_data_path + 'poi_with_fips.csv')
