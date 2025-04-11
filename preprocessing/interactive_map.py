from math import pi, atan, sinh
import folium

def tile_to_latlon(tile_x, tile_y, pixel_x=0, pixel_y=0, zoom=13, res=8192):
    world_width_px  = res * (2**zoom) 
    world_height_px = res * (2**zoom)

    Xglobal_px = tile_x*res + pixel_x
    Yglobal_px = tile_y*res + pixel_y

    lon = Xglobal_px / world_width_px * 360.0 - 180.0
    lat = (180.0 / pi) * atan(sinh(pi * (1 - 2 * Yglobal_px / world_height_px)))

    return lat, lon

def show_tile(tile_x, tile_y, zoom_start=13, radius=10):

    lat1, lon1 = tile_to_latlon(tile_x   , tile_y)   #TopLeft
    lat2, lon2 = tile_to_latlon(tile_x+1 , tile_y)   #TopRight
    lat3, lon3 = tile_to_latlon(tile_x+1 , tile_y+1) #BottomRight
    lat4, lon4 = tile_to_latlon(tile_x   , tile_y+1) #BottomLeft

    LatLongs = [[lat1, lon1], [lat2, lon2], [lat3, lon3], [lat4, lon4]]
    Locations = ['TopLeft', 'TopRight', 'BottomRight', 'BottomLeft']

    m = folium.Map(location=[lat1, lon1], zoom_start=zoom_start)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Maps', 
        name='Google Satellite', 
        overlay=True, 
        control=True
    ).add_to(m)

    for i in range(4):
        folium.CircleMarker(
            LatLongs[i], 
            radius=radius, 
            popup=f'{Locations[i]}-tile: {tile_x}_{tile_y}', 
            color='blue', 
            fill=True, 
            fill_color='red'
        ).add_to(m)

    folium.Polygon(
        locations=LatLongs+[LatLongs[0]], # Close the polygon (back to top-left)
        color='purple',
        fill=True,
        fill_color='orange',
        fill_opacity=0.4
    ).add_to(m)

    return m

def show_crop(tile_x, tile_y, crop, m=None, zoom_start=13, radius=6):

    x1, y1, x2, y2 = crop
    lat1, lon1 = tile_to_latlon(tile_x, tile_y, x1, y1)  #TopLeft
    lat2, lon2 = tile_to_latlon(tile_x, tile_y, x1, y2)  #TopRight
    lat3, lon3 = tile_to_latlon(tile_x, tile_y, x2, y2) #BottomRight
    lat4, lon4 = tile_to_latlon(tile_x, tile_y, x2, y1) #BottomLeft

    LatLongs = [[lat1, lon1], [lat2, lon2], [lat3, lon3], [lat4, lon4]]
    Locations = ['TopLeft', 'TopRight', 'BottomRight', 'BottomLeft']

    if not m: m = folium.Map(location=[lat1, lon1], zoom_start=zoom_start) 
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Maps', 
        name='Google Satellite', 
        overlay=True, 
        control=True
    ).add_to(m)

    for i in range(4):
        folium.CircleMarker(
            LatLongs[i], 
            radius=radius, 
            popup=f'{Locations[i]}-crop: {crop}', 
            color='green', 
            fill=True, 
            fill_color='yellow'
        ).add_to(m)

    folium.Polygon(
        locations=LatLongs+[LatLongs[0]], # Close the polygon (back to top-left)
        color='black',
        fill=True,
        fill_color='white',
        fill_opacity=0.4
    ).add_to(m)

    return m