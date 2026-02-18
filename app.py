# app.py
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run 
import pandas as pd
from src.constants import APP_HOST, APP_PORT
from typing import Optional
from src.logger import logging

# Import your prediction classes (adjust import path if needed)
from src.pipeline.prediction_pipeline import HousingData, HousingDataPredictor

app = FastAPI(
    title="Bengaluru House Price Prediction API",
    description="Predict house prices in Bengaluru based on key features",
    version="1.0.0"
)

# Mount static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder
templates = Jinja2Templates(directory="templates")

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the housing-related attributes expected from the form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.area_type = None
        self.availability = None
        self.location = None
        self.size = None
        self.society = None
        self.total_sqft = None
        self.bath = None
        self.balcony = None
        
    async def get_housing_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.area_type = form.get("area_type")
        self.availability = form.get("availability")
        self.location = form.get("location")
        self.size = form.get("size")
        self.society = form.get("society")
        self.total_sqft = form.get("total_sqft")
        self.bath = form.get("bath")
        self.balcony = form.get("balcony")
        
    # Route to render the main page with the form
    @app.get("/", tags = ["authentication"])
    async def index(request: Request):
        """
        Renders the main HTML form page for housing data input.
        """
        return templates.TemplateResponse(
            "index.html", {"request": request, "context": "Rendering"}
        )
        
        
    # Route to handle form submission and make predictions
    @app.post("/")
    async def predictRouteClient(request: Request):
        """
        Endpoint to receive form data, process it, and make a prediction.
        """
        try:
            form = DataForm(request)
            await form.get_housing_data()
            
            housing_data = HousingData(
                area_type= form.area_type,
                availability= form.availability,
                location= form.location,
                size= form.size,
                society= form.society,
                total_sqft= form.total_sqft,
                bath= form.bath,
                balcony= form.balcony
            )
            
            # Convert form data into a DataFrame for the model
            housing_df = housing_data.get_housing_input_data_frame()
            logging.info(f"The dataframe is : {housing_df}")
            
            # Initialize the prediction pipeline
            model_predictor = HousingDataPredictor()
            
            # Make a prediction and retrieve the result
            value = model_predictor.predict(dataframe= housing_df)[0]
            value = f"{value:.2f} Lakhs"
            
            # Render the same HTML page with the prediction result
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "prediction": value}
            )
            
        except Exception as e:
            return {"status": False, "error": f"{e}"}
      
# Main entry point to start the FastAPI server        
if __name__ == "__main__":
    app_run(app, host = APP_HOST, port = APP_PORT)              