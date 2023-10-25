import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
import mysql.connector
import os
from langchain.llms import OpenAI
import python_weather
import requests
import asyncio
from bs4 import BeautifulSoup
from serpapi import GoogleSearch  # Import the GoogleSearch class


# Load your machine learning models and class names here
def load_models():
    tomato_model = tf.keras.models.load_model('CNN_model_tomatoes')
    pepper_model = tf.keras.models.load_model('pepper-model.h5')
    potato_model = tf.keras.models.load_model('potatoed_disease_classifier')
    pest_model = tf.keras.models.load_model('pest_detection_model2.h5')
    return {'tomato': tomato_model, 'pepper': pepper_model, 'potato': potato_model, 'pest': pest_model}


def get_weather(city_name):
    search_query = f"Weather in {city_name} Nigeria"
    google_url = f"https://www.google.com/search?q={search_query}"

    # Send a GET request to Google
    response = requests.get(google_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        weather_data = soup.find("div", {"class": "BNeawe iBp4i AP7Wnd"}).get_text()
        st.write(f"Weather in {city_name}: {weather_data}")
    else:
        st.write("Failed to fetch weather data. Please check the city name or your internet connection.")


# Function to fetch weather data using python_weather library
async def get_python_weather(city_name):
    async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
        try:
            weather = await client.get(city_name)
            st.write(f"Weather in {city_name}: {weather.current.temperature}°C")
        except Exception as e:
            st.write(f"Failed to fetch weather data: {e}")


def load_class_names():
    return {
        'tomato': ["Healthy", "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot",
                   "Spider Mites", "Target Spot", "Mosaic Virus", "Yellow Leaf Curl Virus"],
        'pepper': ['Bacterial Spot', 'Healthy'],
        'potato': ["Early Blight", "Late Blight", "Healthy"],
        'pest': ['ants', 'aphids', 'bees', 'beetle', 'caterpillar', 'earwig',
                 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil', 'worm']
    }


# Load your database connection function and farm-related functions
def connect_to_database():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="abraham23@",
        database="farmer2"
    )
# Function to retrieve farm activity data by FarmerID
# Function to insert crop data
# Function to insert crop data


def insert_crop_data(crop_name, planting_date, harvest_date, crop_yield):
    db_connection = connect_to_database()
    cursor = db_connection.cursor()
    cursor.callproc("InsertCrop", (crop_name, planting_date, harvest_date, crop_yield))
    db_connection.commit()
    cursor.close()


# Function to retrieve crop data for analysis
# Function to retrieve crop data for analysis
def get_crop_data_for_analysis():
    db_connection = connect_to_database()
    cursor = db_connection.cursor()
    cursor.execute("SELECT CropName, PlantingDate, HarvestDate, Yield AS crop_yield FROM Crops")

    data = cursor.fetchall()
    cursor.close()
    df = pd.DataFrame(data, columns=["Crop Name", "Planting Date", "Harvest Date", "Yield"])
    return df

# Define the Streamlit app title and page icon

# Apply custom CSS for styling
# Define the Streamlit app title and page icon

# Define the Streamlit app title and page icon


st.set_page_config(
    page_title="Green ThumbP: Growing Strong",
    page_icon="🌱"
)

# Apply custom CSS for styling the background color and text color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #c6e48b !important;  /* Background color (light green) */
    }
    .stMarkdown, .stText, .stAlert {
        color: #194d22 !important;  /* Text color (darker green) */
    }
    .stMarkdown h1 {
        font-size: 36px;  /* Header font size */
        text-align: center;  /* Center-align header text */
    }
    .stMarkdown h2 {
        color: #194d22 !important;  /* Subheader text color (darker green) */
    }
    .stApp h1, .stApp h2 {
        color: #194d22 !important;  /* Header and subheader text color (darker green) */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Display the app title and header
st.markdown("# Green ThumbP: Growing Strong")

# Add a decorative line
st.markdown("<hr style='border: 2px solid #194d22;'>", unsafe_allow_html=True)
# Main Streamlit app


def main():
    models = load_models()
    class_names = load_class_names()

    st.sidebar.header("Select Functionality")
    app_mode = st.sidebar.selectbox(
        "Choose the functionality you want to use",
        (
            "Home", "Getting Started", "Tomato Disease Classifier", "Pepper Disease Classifier",
            "Potato Disease Classifier", "Pest Detection",
            "Crop Rotation Advisor", "Farm Animal Vaccination Guide",
            "Question Answering",  # Added "Question Answering" as an option
            "Record Input",
            "Retrieve Crop Data for Analysis", "Check Weather", "Google Search", "Feedback"


        )
    )

    if app_mode == "Home":
        st.title("Farm Management and Plant Health Advisor")
        st.write("Welcome to the Farm Management and Plant Health Advisor!")
        st.write("Select a functionality from the sidebar.")

    elif app_mode == "Tomato Disease Classifier":
        st.header("Tomato Disease Classifier")
        uploaded_file = st.file_uploader("Upload an image of a tomato leaf", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                img = np.array(image)
                img = tf.image.resize(img, (256, 256))
                img = np.expand_dims(img, axis=0)
                img = img / 255.0

                predictions = models['tomato'].predict(img)
                predicted_class = class_names['tomato'][np.argmax(predictions[0])]
                confidence = np.max(predictions[0])

                st.subheader("Prediction Results:")
                st.write(f"Class: {predicted_class}")
                st.write(f"Confidence: {confidence:.2f}")
    elif app_mode == "Pepper Disease Classifier":

        st.header("Pepper Disease Classifier")

        uploaded_file = st.file_uploader("Upload an image of a pepper leaf", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            image = image.resize((128, 128))  # Resize the image to 128x128 pixels

            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                img = np.array(image)

                img = img / 255.0  # Normalize the image

                img = tf.image.resize(img, (128, 128))  # Resize to match the model's input shape

                img = np.expand_dims(img, axis=0)

                predictions = models['pepper'].predict(img)

                predicted_class = class_names['pepper'][np.argmax(predictions[0])]

                confidence = np.max(predictions[0])

                st.subheader("Prediction Results:")

                st.write(f"Class: {predicted_class}")

                st.write(f"Confidence: {confidence:.2f}")
    elif app_mode == "Potato Disease Classifier":
        st.header("Potato Disease Classifier")
        uploaded_file = st.file_uploader("Upload an image of a potato leaf", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                img = np.array(image)
                img = tf.image.resize(img, (256, 256))
                img = np.expand_dims(img, axis=0)
                img = img / 255.0

                predictions = models['potato'].predict(img)
                predicted_class = class_names['potato'][np.argmax(predictions[0])]
                confidence = np.max(predictions[0])

                st.subheader("Prediction Results:")
                st.write(f"Class: {predicted_class}")
                st.write(f"Confidence: {confidence:.2f}")

    elif app_mode == "Pest Detection":
        st.header("Pest Detection")
        uploaded_image = st.file_uploader("Choose an image for pest detection...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Detect Pest"):
                img = np.array(image)
                img = tf.image.resize(img, (128, 128))
                img = tf.expand_dims(img, axis=0) / 255.0

                predictions = models['pest'].predict(img)
                predicted_class = class_names['pest'][np.argmax(predictions[0])]
                confidence = np.max(predictions[0])

                st.subheader("Pest Detection Results:")
                st.write(f"Pest: {predicted_class}")
                st.write(f"Confidence: {confidence:.2f}")


    elif app_mode == "Crop Rotation Advisor":
        st.header("Crop Rotation Advisor")
        st.write("This tool helps you find crop rotation options based on the season and soil type.")

        # Define the data
        data = [
            {"Season": "Rainy Season", "Soil Type": "Loamy Soil", "Months": "April - October",
             "Crops": "Maize, Okra, Peppers, Tomatoes, Eggplant, Watermelon, Cucumber, Okro, Sorghum, Cowpeas"},
            {"Season": "Rainy Season", "Soil Type": "Sandy Soil", "Months": "April - October",
             "Crops": "Vegetables, Maize, Tomato, Amaranth, Pepper, Beans, Yam"},
            {"Season": "Rainy Season", "Soil Type": "Clayey Soil", "Months": "April - October",
             "Crops": "Eggplant, Okro, Cassava, Millet, Yam, Pumpkin, Maize"},
            {"Season": "Rainy Season", "Soil Type": "Organic Matter", "Months": "April - October",
             "Crops": "Maize, Cassava, Tomatoes, Peppers, Beans, Cowpeas"},
            {"Season": "Dry Season", "Soil Type": "Loamy Soil", "Months": "November - March",
             "Crops": "Maize, Cassava, Tomatoes, Peppers, Beans, Cowpeas"},
            {"Season": "Dry Season", "Soil Type": "Sandy Soil", "Months": "November - March",
             "Crops": "Maize, Tomatoes, Okra, Watermelon, Sweet Potatoes"},
            {"Season": "Dry Season", "Soil Type": "Clayey Soil", "Months": "November - March",
             "Crops": "Yam, Plantains, Cocoyam, Cassava, Sweet Potatoes"},
            {"Season": "Dry Season", "Soil Type": "Organic Matter", "Months": "November - March",
             "Crops": "Maize, Cassava, Tomatoes, Peppers, Beans, Cowpeas"},
            {"Season": "Rainy Season", "Soil Type": "Loamy Soil", "Months": "May - November",
             "Crops": "Maize, Okra, Peppers, Tomatoes, Eggplant, Watermelon, Cucumber, Okro, Sorghum, Cowpeas"},
            {"Season": "Rainy Season", "Soil Type": "Sandy Soil", "Months": "May - November",
             "Crops": "Vegetables, Maize, Tomato, Amaranth, Pepper, Beans, Yam"},
            {"Season": "Rainy Season", "Soil Type": "Clayey Soil", "Months": "May - November",
             "Crops": "Eggplant, Okro, Cassava, Millet, Yam, Pumpkin, Maize"},
            {"Season": "Rainy Season", "Soil Type": "Organic Matter", "Months": "May - November",
             "Crops": "Maize, Cassava, Tomatoes, Peppers, Beans, Cowpeas"},
        ]

        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Define options for the dropdowns
        seasons = df["Season"].unique()
        soil_types = df["Soil Type"].unique()

        # Create dropdowns for selecting Season and Soil Type
        selected_season = st.selectbox("Select Season", seasons)
        selected_soil_type = st.selectbox("Select Soil Type", soil_types)

        # Filter the data based on the selected Season and Soil Type
        filtered_data = df[(df["Season"] == selected_season) & (df["Soil Type"] == selected_soil_type)]

        # Display the filtered data
        st.subheader("Crop Rotation Options:")
        st.write("Possible crops to plant:", ", ".join(filtered_data["Crops"]))

        # Footer and acknowledgments
        st.sidebar.text("Powered by OpenAI GPT-3")
        st.sidebar.text("Data sources: Local agricultural experts")



    elif app_mode == "Farm Animal Vaccination Guide":

        data = [
            {"Animal": "Cattle", "Disease": "Foot and Mouth Disease", "Vaccine Name": "Ovac FMD",
             "Dosage": "Varies by vaccine"},
            {"Animal": "Cattle", "Disease": "Brucellosis", "Vaccine Name": "RB-51", "Dosage": "Varies by vaccine"},
            {"Animal": "Sheep", "Disease": "Foot and Mouth Disease", "Vaccine Name": "Ovac FMD",
             "Dosage": "Varies by vaccine"},
            {"Animal": "Sheep", "Disease": "Clostridial Diseases", "Vaccine Name": "Covexin",
             "Dosage": "Varies by vaccine"},
            {"Animal": "Goats", "Disease": "Foot and Mouth Disease", "Vaccine Name": "Ovac FMD",
             "Dosage": "Varies by vaccine"},
            {"Animal": "Goats", "Disease": "Clostridial Diseases", "Vaccine Name": "Covexin",
             "Dosage": "Varies by vaccine"},
            {"Animal": "Poultry (Chickens)", "Disease": "Newcastle Disease", "Vaccine Name": "Newcastle B1 Vaccine",
             "Dosage": "Varies by vaccine"},
            {"Animal": "Poultry (Chickens)", "Disease": "Avian Influenza", "Vaccine Name": "Nobilis Influenza H5N1",
             "Dosage": "Varies by vaccine"},
            {"Animal": "Poultry (Chickens)", "Disease": "Gumboro Disease", "Vaccine Name": "Gumboro IBDV Vaccine",
             "Dosage": "Varies by vaccine"},
            {"Animal": "Poultry (Chickens)", "Disease": "Infectious Bronchitis", "Vaccine Name": "IB Vaccine",
             "Dosage": "Varies by vaccine"},
            {"Animal": "Swine (Pigs)", "Disease": "Foot and Mouth Disease", "Vaccine Name": "Ovac FMD",
             "Dosage": "Varies by vaccine"},
            {"Animal": "Swine (Pigs)", "Disease": "Swine Fever", "Vaccine Name": "Suvaxyn",
             "Dosage": "Varies by vaccine"},
            {"Animal": "Horses", "Disease": "Rabies", "Vaccine Name": "Raboral V-RG", "Dosage": "Varies by vaccine"}
        ]

        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        st.markdown("## Farm Animal Vaccination Guide")
        st.write(
            "Select a farm animal, and get information about common "
            "diseases they are vaccinated against, along with vaccine names and dosages.")

        # Define the farm animal options
        farm_animals = ["Cattle", "Sheep", "Goats", "Poultry (Chickens)", "Swine (Pigs)", "Horses"]

        # Create a dropdown for selecting a farm animal
        selected_animal = st.selectbox("Select a Farm Animal", farm_animals)

        # Filter the data based on the selected farm animal
        filtered_data = df[df["Animal"] == selected_animal]

        # Display the filtered data
        st.write("Common Diseases and Vaccines for", selected_animal)
        st.dataframe(filtered_data)

        # Footer and acknowledgments
        st.sidebar.text("Powered by OpenAI GPT-3")
        st.sidebar.text("Data sources: Local veterinary professionals")


    elif app_mode == "Getting Started":

        st.markdown("# Getting Started with Farming")

        st.markdown("If you're new to farming, here are some essential steps to get started.")

        # Step 1: Buying Seeds

        st.markdown("## Step 1: Buying Seeds")

        st.markdown("To start farming, you'll need quality seeds for your crops. Here are some tips:")

        st.markdown("1. Research the crops suitable for your region and climate.")

        st.markdown("2. Choose reputable seed suppliers or local markets.")

        st.image("planting.png", caption="Sample Seed Packet", use_column_width=True)

        # Step 2: Planting

        st.markdown("## Step 2: Planting")

        st.markdown("Planting is a crucial step in farming. Here's what you need to know:")

        st.markdown("1. Prepare the soil by clearing debris and adding compost (manure).")

        st.markdown("2. Follow the recommended planting depth and spacing for each crop.")

        st.image("planting.png", caption="Planting Crops", use_column_width=True)

        # Step 4: Fertilization

        st.markdown("## Step 3: Fertilization")

        st.markdown("Fertilization is crucial for healthy plant growth. Here's what you need to know:")

        st.markdown("1. Choose the right type of fertilizer based on your crop's nutrient requirements.")

        st.markdown("2. Follow recommended fertilizer application rates and timings.")

        st.image("fertilizer.jpg", caption="Fertilizing Crops", use_column_width=True)

        # Step 5: Watering

        st.markdown("## Step 4: Watering")

        st.markdown("Proper watering is essential for plant health. Here's some watering advice:")

        st.markdown("1. Water your crops regularly, but avoid overwatering, which can lead to root rot.")

        st.markdown("2. Use a watering schedule that matches your crop's water needs.")

        st.image("Watering.jpg", caption="Watering Crops", use_column_width=True)

        # Step 6: Pest Control

        st.markdown("## Step 5: Pest Control")

        st.markdown("Protect your crops from pests. Here are some pest control tips:")

        st.markdown("1. Identify common pests in your region and monitor your crops regularly.")

        st.markdown("2. Use organic or chemical pest control methods as needed.")

        st.image("pest.jpg", caption="Pest Control", use_column_width=True)

        # Step 3: Harvesting (Last Step)

        st.markdown("## Step 6: Harvesting")

        st.markdown("Harvesting is the final step in the farming process. Here are some tips:")

        st.markdown("1. Harvest crops at the right time. Different crops have different maturity periods.")

        st.markdown("2. Use appropriate tools for harvesting to avoid damage to the crops.")

        st.image("harvesting.png", caption="Harvesting Crops", use_column_width=True)
    elif app_mode == "Question Answering":

        os.environ['OPENAI_API_KEY'] = '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'  # Set the OpenAI API Key

        st.markdown("## Question Answering")

        # Allow the user to enter a question

        user_question = st.text_input("Ask a question:", "")

        if st.button("Get Answer"):

            if user_question:

                try:

                    # Use OpenAI to generate a response for the user's question

                    llm = OpenAI(temperature=0.6)

                    openai_response = llm(user_question)

                    st.subheader("OpenAI Response:")

                    st.write(f"Question: {user_question}")

                    st.write(f"Response: {openai_response}")
                except Exception as e:

                    st.write("Error processing the question:", str(e))

            else:

                st.write("Please enter a question.")
    elif app_mode == "Check Weather":

        st.header("Check Weather in Nigeria")

        st.write("Use this tool to check the weather in a specific Nigerian city.")

        city_name = st.text_input("Enter the city name in Nigeria:")

        if st.button("Get Weather"):

            if city_name:

                asyncio.run(get_python_weather(city_name))

            else:

                st.warning("Please enter a city name.")

    elif app_mode == "Google Search":
        st.header("Google Search")

        # User input for the search query
        user_question = st.text_input("Ask a question or enter a search query:")

        if st.button("Search"):
            # Check if the user entered a question/query
            if not user_question:
                st.warning("Please enter a question or query.")
            else:
                # Perform a Google search using serpapi.GoogleSearch
                search_params = {
                    "q": user_question,
                    "api_key": "bda5642fb7edb79b6af75fe1ac504b8d302bef4cb8b6713a65c511e74a2ae505",
                    # Add other parameters like location, etc. as needed
                }

                search = GoogleSearch(search_params)
                search_results = search.get_dict()

                st.subheader("SERPAPI (GOOGLE INFO) Search Results:")

                # Display the top search result (the first organic result)
                if search_results and "organic_results" in search_results:
                    top_result = search_results["organic_results"][0]
                    st.write(top_result.get("title"))
                    st.write(top_result.get("snippet"))
                else:
                    st.warning("No search results found for the query.")
    elif app_mode == "Record Input":

        st.header("Record Crop Data")

        # User input form for recording crop data

        with st.form("record_crop_data"):

            crop_name = st.text_input("Crop Name")

            planting_date = st.date_input("Planting Date")

            harvest_date = st.date_input("Harvest Date")

            crop_yield = st.number_input("Yield (kg)")

            # Add input fields for other data columns as needed

            submitted = st.form_submit_button("Submit")

            if submitted:

                # Insert the crop data using the stored function

                insert_crop_data(crop_name, planting_date, harvest_date, crop_yield)

                st.success("Crop data recorded successfully!")

                # Optionally, display the recorded data to the user

                st.subheader("Recorded Crop Data")

                st.write("Crop Name:", crop_name)

                st.write("Planting Date:", planting_date)

                st.write("Harvest Date:", harvest_date)

                st.write("Yield (kg):", crop_yield)

                # Display other recorded data as needed
            else:

                st.write("Fill out the form and click 'Submit' to record crop data.")
    elif app_mode == "Retrieve Crop Data for Analysis":
        st.header("Retrieve Crop Data for Analysis")

        # Retrieve crop data
        crop_data = get_crop_data_for_analysis()

        if not crop_data.empty:
            st.write("Here is the crop data for analysis:")
            st.write(crop_data)

            # Assuming you have already retrieved crop data as "crop_data"

            import matplotlib.pyplot as plt

            # Perform data analysis
            average_yield = crop_data['Yield'].mean()
            max_yield = crop_data['Yield'].max()
            min_yield = crop_data['Yield'].min()

            # Create a histogram of yields
            plt.figure(figsize=(8, 6))
            plt.hist(crop_data['Yield'], bins=20, color='skyblue')
            plt.xlabel('Yield')
            plt.ylabel('Frequency')
            plt.title('Yield Distribution')
            plt.grid(True)

            # Display statistics
            st.subheader("Crop Data Analysis")
            st.write(f"Average Yield: {average_yield:.2f}")
            st.write(f"Maximum Yield: {max_yield}")
            st.write(f"Minimum Yield: {min_yield}")

            # Display the histogram
            st.subheader("Yield Distribution")
            st.pyplot(plt)

            # You can perform data analysis or visualization here
            # For example, using pandas and matplotlib

        else:
            st.write("No crop data available for analysis.")
    # Your Streamlit app code here...

    # Create an elif statement to handle feedback link
    elif app_mode == "Feedback":
        st.header("Feedback")
        st.write("We value your feedback. Please send us your comments, questions, or suggestions.")

        # Add an email link
        email = "abrahamsunday23@gmail.com"
        st.markdown(f"Send an email to: [{email}](mailto:{email})")


# Run the Streamlit app
if __name__ == "__main__":
    main()
