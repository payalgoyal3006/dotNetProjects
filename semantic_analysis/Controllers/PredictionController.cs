using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.Data;
using SemanticML.Model;


namespace semantic.Controllers
{
    public class PredictionController : Controller
    {
        [HttpGet]
        public IActionResult Prediction()
        {
            return View();
        }

        [HttpPost]
        public ActionResult Prediction(ModelInput input)
        {
            ViewBag.Result = "";
            var predictions = ConsumeModel.Predict(input);
            ViewBag.Result = predictions;

            ViewData["Feedbacktext"] = input.Feedbacktext;

            return View();
        }
    }
}
