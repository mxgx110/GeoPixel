import json
import numpy as np   
from PIL import Image
import random
from shapely.geometry import Point, LineString, Polygon, box

def crop_image_and_labels(image_path, json_path, bbox):
    img = Image.fromarray(np.load(image_path)) 
    W, H = img.size
    x1, y1, x2, y2 = bbox
    crop = img.crop((x1, y1, x2, y2))
    crop_box = box(x1, y1, x2, y2)

    with open(json_path) as f:
        data = json.load(f)
    new_data = {}

    for key, items in data.items():
        new_items = []
        for item in items:
            g = item["Geometry"]
            t = g["Type"]
            pt = g['Point'] if 'Point' in g.keys() else None
            pl = g['Polyline'] if 'Polyline' in g.keys() else None
            pg = g["Polygon"]
            g2 = {"Type": t, "Point": None, "Polyline": None, "Polygon": None}

            if pt:
                p = Point(pt)
                if crop_box.contains(p):
                    g2["Point"] = [p.x - x1, p.y - y1]

            if pl:
                line = LineString(pl)
                clipped = line.intersection(crop_box)
                if not clipped.is_empty:
                    if clipped.geom_type == "LineString":
                        g2["Polyline"] = [[p[0] - x1, p[1] - y1] for p in clipped.coords]
                    elif clipped.geom_type == "MultiLineString":
                        g2["Polyline"] = [[p[0] - x1, p[1] - y1] for l in clipped.geoms for p in l.coords]

            if pg:
                new_polys = []
                for poly in pg:
                    polygon = Polygon(poly)
                    clipped = polygon.intersection(crop_box)
                    if not clipped.is_empty:
                        if clipped.geom_type == "Polygon":
                            new_polys.append([[p[0] - x1, p[1] - y1] for p in list(clipped.exterior.coords)])
                        elif clipped.geom_type == "MultiPolygon":
                            for p in clipped.geoms:
                                new_polys.append([[pt[0] - x1, pt[1] - y1] for pt in list(p.exterior.coords)])
                if new_polys:
                    g2["Polygon"] = new_polys

            if g2["Point"] or g2["Polyline"] or g2["Polygon"]:
                new_items.append({"Geometry": g2, "Properties": item.get("Properties")})
        new_data[key] = new_items

    return crop, new_data, data

color = lambda: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))