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
using ArcGIS.Desktop.Layouts;
using ArcGIS.Desktop.Mapping;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WaterCorpWeedMonitor
{
    internal class Help : Button
    {
        protected override void OnClick()
        {
            try
            {
                // open system browser to github page
                string url = "https://github.com/lewistrotter/WaterCorpWeedMonitor";
                var process = new ProcessStartInfo(url) { UseShellExecute = true };
                System.Diagnostics.Process.Start(process);
            }
            catch (Exception e)
            {
                MessageBox.Show("Could not reach GitHub page.");
                System.Diagnostics.Debug.WriteLine(e.Message);
            }
        }
    }
}
