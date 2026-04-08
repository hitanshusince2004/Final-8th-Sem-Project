"""
Glacier Melting Detection - GEE Utility Functions
=================================================
Reusable helpers for cloud masking, spectral index computation,
SAR preprocessing, and LST derivation.
"""

import ee


# ══════════════════════════════════════════════════════════════════════════════
# CLOUD MASKING
# ══════════════════════════════════════════════════════════════════════════════

def mask_s2_clouds(image):
    """
    Mask clouds and cirrus in Sentinel-2 SR using the SCL (Scene Classification)
    band and the QA60 bitmask band.
    Returns the image with cloud pixels set to NoData and the 'cloud_score' property.
    """
    scl = image.select("SCL")

    # SCL classes to EXCLUDE: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=cirrus
    cloud_shadow = scl.eq(3)
    cloud_med    = scl.eq(8)
    cloud_high   = scl.eq(9)
    cirrus       = scl.eq(10)

    # Additional QA60 bitmask (bits 10 & 11)
    qa60         = image.select("QA60")
    qa_cloud     = qa60.bitwiseAnd(1 << 10).neq(0)
    qa_cirrus    = qa60.bitwiseAnd(1 << 11).neq(0)

    mask = (cloud_shadow
            .Or(cloud_med)
            .Or(cloud_high)
            .Or(cirrus)
            .Or(qa_cloud)
            .Or(qa_cirrus)
            .Not())

    return (image
            .updateMask(mask)
            .divide(10000)          # scale to [0, 1] reflectance
            .copyProperties(image, ["system:time_start", "system:index"]))


def mask_landsat_clouds(image):
    """
    Mask clouds and cloud shadows in Landsat 8/9 Collection 2 using QA_PIXEL band.
    Also applies the scale factors for SR and ST bands.
    """
    qa        = image.select("QA_PIXEL")
    cloud     = qa.bitwiseAnd(1 << 3).neq(0)
    cloud_shd = qa.bitwiseAnd(1 << 4).neq(0)
    mask      = cloud.Or(cloud_shd).Not()

    # Apply Collection 2 scale factors
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal_bands = image.select("ST_B10").multiply(0.00341802).add(149.0)

    return (image
            .addBands(optical_bands, overwrite=True)
            .addBands(thermal_bands, overwrite=True)
            .updateMask(mask)
            .copyProperties(image, ["system:time_start", "system:index"]))


def mask_s1_border_noise(image):
    """
    Remove Sentinel-1 GRD border and thermal noise.
    Keeps only pixels with VV and VH values in a reasonable dB range.
    """
    edge    = image.lt(-30.0)
    mask_vv = image.select("VV").gt(-30)
    mask_vh = image.select("VH").gt(-30)
    return image.updateMask(mask_vv.And(mask_vh).And(edge.select("VV").Not()))


# ══════════════════════════════════════════════════════════════════════════════
# SPECTRAL INDICES
# ══════════════════════════════════════════════════════════════════════════════

def compute_ndsi_s2(image):
    """
    NDSI (Normalised Difference Snow Index) from Sentinel-2.
    NDSI = (Green - SWIR1) / (Green + SWIR1)
    Values > 0.40 strongly indicate snow/ice cover.
    """
    ndsi = image.normalizedDifference(["B3", "B11"]).rename("NDSI")
    return image.addBands(ndsi)


def compute_ndsi_landsat(image):
    """NDSI from Landsat 8/9 (Green = B3, SWIR1 = B6)."""
    ndsi = image.normalizedDifference(["SR_B3", "SR_B6"]).rename("NDSI")
    return image.addBands(ndsi)


def compute_ndwi_s2(image):
    """
    NDWI (Normalised Difference Water Index) - McFeeters (1996).
    NDWI = (Green - NIR) / (Green + NIR)
    Highlights open water bodies (meltwater lakes, proglacial lakes).
    """
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return image.addBands(ndwi)


def compute_ndwi_landsat(image):
    """NDWI from Landsat 8/9."""
    ndwi = image.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
    return image.addBands(ndwi)


def compute_ndvi_s2(image):
    """NDVI - used as a supporting feature to distinguish glacier from vegetation."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def compute_ndvi_landsat(image):
    ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    return image.addBands(ndvi)


def compute_evi(image):
    """
    Enhanced Vegetation Index (EVI).
    EVI = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
    """
    nir = image.select("B8")
    red = image.select("B4")
    blue = image.select("B2")
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {"NIR": nir, "RED": red, "BLUE": blue}
    ).rename("EVI")
    return image.addBands(evi)


def compute_savi(image):
    """
    Soil Adjusted Vegetation Index (SAVI).
    SAVI = ((NIR - Red) / (NIR + Red + 0.5)) * (1.5)
    """
    nir = image.select("B8")
    red = image.select("B4")
    savi = image.expression(
        "((NIR - RED) / (NIR + RED + 0.5)) * 1.5",
        {"NIR": nir, "RED": red}
    ).rename("SAVI")
    return image.addBands(savi)


def compute_bai(image):
    """
    Bare Area Index (BAI) - highlights exposed rock/debris on glaciers.
    BAI = 1 / ((0.1 - Red)**2 + (0.06 - NIR)**2)
    """
    red = image.select("B4")
    nir = image.select("B8")
    bai = (ee.Image(1.0)
           .divide(
               (red.subtract(0.1)).pow(2)
               .add((nir.subtract(0.06)).pow(2))
           )
           .rename("BAI"))
    return image.addBands(bai)


def compute_sar_ratio(image):
    """
    VV/VH ratio for Sentinel-1. Useful for discriminating ice types.
    Works in linear scale; convert dB -> linear first.
    """
    vv_lin  = ee.Image(10).pow(image.select("VV").divide(10)).rename("VV_lin")
    vh_lin  = ee.Image(10).pow(image.select("VH").divide(10)).rename("VH_lin")
    ratio   = vv_lin.divide(vh_lin).rename("VV_VH_ratio")
    return image.addBands([vv_lin, vh_lin, ratio])


def compute_sar_texture(image):
    """
    GLCM texture features from SAR VV band.
    Contrast + entropy help distinguish rough vs smooth ice surfaces.
    """
    vv_int  = image.select("VV").multiply(1000).int32()
    glcm    = vv_int.glcmTexture(size=3)
    contrast = glcm.select("VV_contrast").rename("SAR_contrast")
    entropy  = glcm.select("VV_ent").rename("SAR_entropy")
    return image.addBands([contrast, entropy])


# ══════════════════════════════════════════════════════════════════════════════
# LAND SURFACE TEMPERATURE
# ══════════════════════════════════════════════════════════════════════════════

def compute_lst_celsius(image):
    """
    Convert Landsat 8/9 ST_B10 (already in Kelvin after scaling) to Celsius.
    Also computes emissivity-corrected LST using NDVI-based emissivity estimation.
    """
    # ST_B10 is in Kelvin after Collection 2 scaling
    lst_kelvin = image.select("ST_B10")
    lst_c      = lst_kelvin.subtract(273.15).rename("LST_Celsius")

    # Emissivity correction using NDVI
    ndvi       = image.select("NDVI")
    fvc        = ndvi.subtract(0.2).divide(0.5 - 0.2).clamp(0, 1)   # Fractional Veg Cover
    emissivity = fvc.multiply(0.004).add(0.986).rename("Emissivity")

    return image.addBands([lst_c, emissivity])


# ══════════════════════════════════════════════════════════════════════════════
# GLACIER MASK GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def create_glacier_mask(image, ndsi_threshold=0.40):
    """
    Generate a binary glacier mask from NDSI.
      1 = glacier / snow / ice
      0 = non-glacier
    Optionally refined with NDWI to exclude open water.
    """
    import ee
    
    ndsi      = image.select("NDSI")
    glacier   = ndsi.gt(ndsi_threshold).uint8().rename("glacier_mask")

    # Exclude pixels classified as open water (using NDWI if present)
    # We check bandNames server-side or just use it if we know it exists
    # For Sentinel-2 composite, NDWI is added in previous map step.
    has_ndwi  = image.bandNames().contains("NDWI")
    
    # Use ee.Algorithms.If for server-side conditional logic
    def with_ndwi():
        not_water = image.select("NDWI").lt(0.30)
        return glacier.And(not_water).uint8().rename("glacier_mask")
        
    glacier = ee.Image(ee.Algorithms.If(has_ndwi, with_ndwi(), glacier))

    return image.addBands(glacier)


def create_multiclass_mask(image):
    """
    4-class segmentation mask:
      0 = land / bare rock / debris
      1 = snow / clean ice
      2 = open water (proglacial / meltwater)
      3 = mixed / debris-covered ice
    """
    ndsi   = image.select("NDSI")
    ndwi   = image.select("NDWI")
    ndvi   = image.select("NDVI")

    snow_ice = ndsi.gt(0.40)                                 # class 1
    water    = ndwi.gt(0.30).And(snow_ice.Not())             # class 2
    debris   = (ndsi.gt(0.10).And(ndsi.lte(0.40))
                .And(ndvi.lt(0.20))
                .And(water.Not()))                           # class 3
    land     = snow_ice.Or(water).Or(debris).Not()           # class 0

    mask = (land.multiply(0)
            .add(snow_ice.multiply(1))
            .add(water.multiply(2))
            .add(debris.multiply(3))
            .rename("multiclass_mask"))

    return image.addBands(mask)


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_s2_composite(aoi, year, season):
    """
    Build a Sentinel-2 seasonal median composite for a given year and season.
    Applies cloud masking and computes all indices.
    """
    import ee
    from config.settings import S2_COLLECTION, S2_CLOUD_COVER

    start = f"{year}-{season['start']}"
    end   = f"{year}-{season['end']}"

    # Handle winter season crossing year boundary
    if season["start"] > season["end"]:
        end = f"{year + 1}-{season['end']}"

    collection = (ee.ImageCollection(S2_COLLECTION)
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_COVER))
                  .map(mask_s2_clouds)
                  .map(compute_ndsi_s2)
                  .map(compute_ndwi_s2)
                  .map(compute_ndvi_s2)
                  .map(compute_evi)
                  .map(compute_savi)
                  .map(compute_bai)
                  .map(create_glacier_mask)
                  .map(create_multiclass_mask))

    composite = collection.median().clip(aoi)
    return composite.set({
        "year":   year,
        "season": "melt" if season["start"] == "05-01" else "winter",
        "system:time_start": ee.Date(start).millis()
    })


def build_s1_composite(aoi, year, season):
    """Build a Sentinel-1 seasonal median composite."""
    import ee
    from config.settings import S1_COLLECTION, S1_MODE

    start = f"{year}-{season['start']}"
    end   = f"{year}-{season['end']}"

    # Handle winter season crossing year boundary
    if season["start"] > season["end"]:
        end = f"{year + 1}-{season['end']}"

    collection = (ee.ImageCollection(S1_COLLECTION)
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .filter(ee.Filter.eq("instrumentMode", S1_MODE))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                  .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
                  .map(mask_s1_border_noise)
                  .map(compute_sar_ratio)
                  .map(compute_sar_texture))

    composite = collection.median().clip(aoi)
    return composite.set({
        "year":   year,
        "season": "melt" if season["start"] == "05-01" else "winter",
    })


def build_landsat_composite(aoi, year, season):
    """Build a Landsat 8/9 seasonal median composite with LST."""
    import ee
    from config.settings import L8_COLLECTION, L9_COLLECTION, LANDSAT_CLOUD

    start = f"{year}-{season['start']}"
    end   = f"{year}-{season['end']}"

    # Handle winter season crossing year boundary
    if season["start"] > season["end"]:
        end = f"{year + 1}-{season['end']}"

    l8 = (ee.ImageCollection(L8_COLLECTION)
          .filterBounds(aoi)
          .filterDate(start, end)
          .filter(ee.Filter.lt("CLOUD_COVER", LANDSAT_CLOUD))
          .map(mask_landsat_clouds))

    l9 = (ee.ImageCollection(L9_COLLECTION)
          .filterBounds(aoi)
          .filterDate(start, end)
          .filter(ee.Filter.lt("CLOUD_COVER", LANDSAT_CLOUD))
          .map(mask_landsat_clouds))

    combined = (
        l8.merge(l9)
        .map(compute_ndsi_landsat)
        .map(compute_ndwi_landsat)
        .map(compute_ndvi_landsat)
        .map(compute_lst_celsius)
        .map(create_glacier_mask)
        .map(create_multiclass_mask)
    )
    composite = combined.median().clip(aoi)
    return composite.set({
        "year":   year,
        "season": "melt" if season["start"] == "05-01" else "winter",
        "system:time_start": ee.Date(start).millis()
    })


def get_dem(aoi):
    """Fetch and clip SRTM DEM, compute slope and aspect."""
    import ee
    from config.settings import DEM_COLLECTION

    dem     = ee.Image(DEM_COLLECTION).clip(aoi).rename("elevation")
    terrain = ee.Terrain.products(dem)
    slope   = terrain.select("slope")
    aspect  = terrain.select("aspect")
    return dem.addBands([slope, aspect])
