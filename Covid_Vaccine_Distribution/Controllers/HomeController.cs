using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.Data;
using Covid_Vaccine_DistributionML.Model;

namespace Covid_Vaccine_Distribution.Controllers
{
    public class HomeController : Controller
    {
        public IActionResult Index(ModelInput input)
        {
            ModelOutput result = ConsumeModel.Predict(input);

            ViewBag.Vaccine = result.Score;
            return View(input);
        }
    }
}
