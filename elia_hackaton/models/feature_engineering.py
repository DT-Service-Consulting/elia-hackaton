# Selecting relevant features, including weather data
features = ["nominalLoad", "heatRunTest_noLoadLosses", "heatRunTest_copperLosses",
            "heatRunTest_ambiantTemperature", "heatRunTest_deltaTopOil",
            "heatRunTest_x", "heatRunTest_y", "heatRunTest_h", "heatRunTest_gradient",
            # Including ambient temperature as weather data
            "load", "heatRunTest_ambiantTemperature"]
target = "hotspotTemperature"