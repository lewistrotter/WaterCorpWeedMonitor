using ArcGIS.Core.CIM;
using ArcGIS.Core.Data;
using ArcGIS.Core.Geometry;
using ArcGIS.Desktop.Catalog;
using ArcGIS.Desktop.Core;
using ArcGIS.Desktop.Editing;
using ArcGIS.Desktop.Extensions;
using ArcGIS.Desktop.Framework;
using ArcGIS.Desktop.Framework.Contracts;
using ArcGIS.Desktop.Framework.Dialogs;
using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Desktop.Internal.Mapping;
using ArcGIS.Desktop.Layouts;
using ArcGIS.Desktop.Mapping;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace WaterCorpWeedMonitor
{
    internal class GraphFractions : Button
    {
        protected async override void OnClick()
        {
            if (MapView.Active != null)
            {
                var mapView = MapView.Active;

                var selectedLayers = mapView.GetSelectedLayers();
                if (selectedLayers != null && selectedLayers.Count == 1)
                {
                    var selectedLayer = selectedLayers.First();
                    if (selectedLayer is RasterLayer)
                    {
                        await QueuedTask.Run(() => {
                            var rasterLayer = selectedLayer as RasterLayer;
                            if (rasterLayer.IsTimeEnabled())
                            {
                                var rasterLyrDef = rasterLayer.GetDefinition();

                                if (rasterLyrDef.Name == "s2_fracs.crf")
                                {
                                    var temporalProfileChart = new CIMChart
                                    {
                                        Name = "WCMonitor Temporal Profile Fractionals",

                                        GeneralProperties = new CIMChartGeneralProperties
                                        {
                                            Title = "Natives vs Weeds vs Other Fractions"
                                        },

                                        Axes = new CIMChartAxis[]
                                        {
                                        new CIMChartAxis
                                        {
                                            Title = "StdTime"
                                        },
                                        new CIMChartAxis
                                        {
                                            Title = "Values"
                                        }
                                        },

                                        //MetaData = 

                                        Series = new CIMChartSeries[]
                                        {
                                        new CIMChartDimensionalProfileSeries
                                        {
                                            UniqueName = "Series1",
                                            Name = "temporalProfile",
                                            MultiSeries = true,

                                            PlotType = ChartDimensionalProfilePlotType.Variables,  // multi vars over time at one location

                                            Fields = new string[] { "StdTime" },
                                            OrderFields = new string[] { "StdTime" },

                                            TimeAggregationType = ChartTimeAggregationType.EqualIntervalsFromStartTime,
                                            TimeIntervalUnits = esriTimeUnits.esriTimeUnitsMonths,
                                            TimeIntervalSize = 1.0,
                                            TrimIncompleteTimeInterval = true,

                                            Variables = new CIMChartDimensionalProfileVariable[]
                                            {
                                                // Natives line
                                                new CIMChartDimensionalProfileVariable
                                                {
                                                    Name = "native",
                                                    Label = "native",
                                                    Symbol = new CIMSymbolReference
                                                    {
                                                        Symbol = new CIMLineSymbol
                                                        {
                                                            SymbolLayers = new CIMSymbolLayer[]
                                                            {
                                                                new CIMSolidStroke
                                                                {
                                                                    Color = new CIMRGBColor { R=46, G=179, B=64, Alpha=50 },
                                                                    Width = 2.0
                                                                }
                                                            }
                                                        }
                                                    }
                                                },

                                                // Weeds line
                                                new CIMChartDimensionalProfileVariable
                                                {
                                                    Name = "weed",
                                                    Label = "weed",
                                                    Symbol = new CIMSymbolReference
                                                    {
                                                        Symbol = new CIMLineSymbol
                                                        {
                                                            SymbolLayers = new CIMSymbolLayer[]
                                                            {
                                                                new CIMSolidStroke
                                                                {
                                                                    Color = new CIMRGBColor { R=194, G=56, B=21, Alpha=50 },
                                                                    Width = 2.0
                                                                }
                                                            }
                                                        }
                                                    }
                                                },

                                                // Other line
                                                new CIMChartDimensionalProfileVariable
                                                {
                                                    Name = "other",
                                                    Label = "other",
                                                    Symbol = new CIMSymbolReference
                                                    {
                                                        Symbol = new CIMLineSymbol
                                                        {
                                                            SymbolLayers = new CIMSymbolLayer[]
                                                            {
                                                                new CIMSolidStroke
                                                                {
                                                                    Color = new CIMRGBColor { R=126, G=127, B=130, Alpha=50 },
                                                                    Width = 2.0
                                                                }
                                                            }
                                                        }
                                                    }
                                                },
                                            }
                                        }
                                        }
                                    };


                                    // Add new chart to layer's existing list of charts (if any exist)
                                    var newTemporalChart = new CIMChart[] { temporalProfileChart };
                                    //var allTemporalChart = (rasterLyrDef == null) ? newTemporalChart : rasterLyrDef.Charts.Concat(newTemporalChart);

                                    // Add CIM chart to layer defintion 
                                    //rasterLyrDef.Charts = allTemporalChart.ToArray();
                                    rasterLyrDef.Charts = newTemporalChart.ToArray();
                                    rasterLayer.SetDefinition(rasterLyrDef);
                                }

                                else if (rasterLyrDef.Name == "s2_ndvi.crf")
                                {
                                    var temporalProfileChart = new CIMChart
                                    {
                                        Name = "WCMonitor Temporal Profile NDVI",

                                        GeneralProperties = new CIMChartGeneralProperties
                                        {
                                            Title = "NDVI"
                                        },

                                        Axes = new CIMChartAxis[]
                                        {
                                        new CIMChartAxis
                                        {
                                            Title = "StdTime"
                                        },
                                        new CIMChartAxis
                                        {
                                            Title = "Values"
                                        }
                                        },

                                        //MetaData = 

                                        Series = new CIMChartSeries[]
                                        {
                                        new CIMChartDimensionalProfileSeries
                                        {
                                            UniqueName = "Series2",
                                            Name = "temporalProfile",
                                            MultiSeries = true,

                                            PlotType = ChartDimensionalProfilePlotType.Variables,  // multi vars over time at one location

                                            Fields = new string[] { "StdTime" },
                                            OrderFields = new string[] { "StdTime" },

                                            TimeAggregationType = ChartTimeAggregationType.EqualIntervalsFromStartTime,
                                            TimeIntervalUnits = esriTimeUnits.esriTimeUnitsMonths,
                                            TimeIntervalSize = 1.0,
                                            TrimIncompleteTimeInterval = true,

                                            Variables = new CIMChartDimensionalProfileVariable[]
                                            {
                                                // Natives line
                                                new CIMChartDimensionalProfileVariable
                                                {
                                                    Name = "ndvi",
                                                    Label = "ndvi",
                                                    Symbol = new CIMSymbolReference
                                                    {
                                                        Symbol = new CIMLineSymbol
                                                        {
                                                            SymbolLayers = new CIMSymbolLayer[]
                                                            {
                                                                new CIMSolidStroke
                                                                {
                                                                    Color = new CIMRGBColor { R=46, G=179, B=64, Alpha=50 },
                                                                    Width = 2.0
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        }
                                    };

                                    // Add new chart to layer's existing list of charts (if any exist)
                                    var newTemporalChart = new CIMChart[] { temporalProfileChart };
                                    //var allTemporalChart = (rasterLyrDef == null) ? newTemporalChart : rasterLyrDef.Charts.Concat(newTemporalChart);

                                    // Add CIM chart to layer defintion 
                                    rasterLyrDef.Charts = newTemporalChart.ToArray(); //allTemporalChart.ToArray();
                                    rasterLayer.SetDefinition(rasterLyrDef);
                                }

                                var x = 0;

                            }
                        });
                    }
                }
            }
        }
    }
}
