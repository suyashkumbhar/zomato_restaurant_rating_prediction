import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import gzip
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='streamlit_app.log',
)

# Log information about the start of the application
logging.info("Streamlit app started.")

# Function to decompress model
def decompress_model():
    try:
        with gzip.open('compressed_regressor_model.joblib.gz', 'rb') as f:
            buffer = BytesIO(f.read())
        return joblib.load(buffer)
    except Exception as e:
        # Log any errors that occur during model loading
        logging.error(f"Error loading model: {str(e)}", exc_info=True)
        raise e

# Log information about loading the model
logging.info("Loading the model...")
model = decompress_model()
logging.info("Model loaded successfully.")

# Function to preprocess input data
def preprocess_input(data, label_encoders=None):
    try:
        # Log information about preprocessing input data
        logging.info("Preprocessing input data...")
        
        # Convert 'Yes'/'No' to 1/0 for binary features
        data['online_orders'] = (data['online_orders'] == 'Yes').astype(int)
        data['book_table'] = (data['book_table'] == 'Yes').astype(int)
        
        # Label encode categorical columns
        columns_to_encode = ['restaurant_location', 'restaurant_type', 'cuisine_types', 'restaurant_service']
        
        if label_encoders is None:
            label_encoders = {}
        
        for column in columns_to_encode:
            if column in data:
                if column not in label_encoders:
                    label_encoders[column] = LabelEncoder()
                    label_encoders[column].fit(data[column])
                data[column] = label_encoders[column].transform(data[column])
        
        # Log information about successful preprocessing
        logging.info("Input data preprocessing successful.")
        
        return data, label_encoders
    except Exception as e:
        # Log any errors that occur during preprocessing
        logging.error(f"Error during input data preprocessing: {str(e)}", exc_info=True)
        raise e

# Function to predict restaurant rating
def predict_rating(features):
    try:
        # Log information about making predictions
        logging.info("Making predictions...")
        
        # Preprocess input features
        features, _ = preprocess_input(features)
        
        # Make prediction using the loaded model
        rating_prediction = model.predict(features)
        
        # Log information about successful predictions
        logging.info("Predictions successful.")
        
        return rating_prediction
    except Exception as e:
        # Log any errors that occur during prediction
        logging.error(f"Error during predictions: {str(e)}", exc_info=True)
        raise e

# Streamlit app
def main():
    try:
        # Page configuration
        st.set_page_config(
            page_title="Restaurant Rating Prediction",
            page_icon="🍔",
            layout="centered",
        )

        # Add background image using custom CSS
        st.markdown(
        """
        <style>
            body {
                background: url('data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7') repeat fixed center center;
                background-size: cover;
            }
        </style>
        """,
        unsafe_allow_html=True,
        )

        # Image header
        st.image("cultural-cuisine.jpg", use_column_width=True)

        # Title and description
        st.title('Welcome to Restaurant Rating Prediction App')
        st.markdown(
            "Enter details about the restaurant, and I will predict the rating for you! 🌟"
        )

        # Collect input features from user
        online_orders = st.selectbox('Online Orders', ['Yes', 'No'])
        book_table = st.selectbox('Book Table', ['Yes', 'No'])
        total_votes = st.number_input('Total Votes', min_value=0, max_value=16832)
        restaurant_location = st.selectbox('Restaurant Location', ['Banashankari', 'Basavanagudi', 'Mysore Road', 'Jayanagar', 'Kumaraswamy Layout', 'Rajarajeshwari Nagar', 'Vijay Nagar', 'Uttarahalli', 'JP Nagar', 'South Bangalore', 'City Market', 'Nagarbhavi', 'Bannerghatta Road', 'BTM', 'Kanakapura Road', 'Bommanahalli', 'CV Raman Nagar', 'Electronic City', 'HSR', 'Marathahalli', 'Sarjapur Road', 'Wilson Garden', 'Shanti Nagar', 'Koramangala 5th Block', 'Koramangala 8th Block', 'Richmond Road', 'Koramangala 7th Block', 'Jalahalli', 'Koramangala 4th Block', 'Bellandur', 'Whitefield', 'East Bangalore', 'Old Airport Road', 'Indiranagar', 'Koramangala 1st Block', 'Frazer Town', 'RT Nagar', 'MG Road', 'Brigade Road', 'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road', 'Shivajinagar', 'Infantry Road', 'St. Marks Road', 'Cunningham Road', 'Race Course Road', 'Commercial Street', 'Vasanth Nagar', 'HBR Layout', 'Domlur', 'Ejipura', 'Jeevan Bhima Nagar', 'Old Madras Road', 'Malleshwaram', 'Seshadripuram', 'Kammanahalli', 'Koramangala 6th Block', 'Majestic', 'Langford Town', 'Central Bangalore', 'Sanjay Nagar', 'Brookefield', 'ITPL Main Road, Whitefield', 'Varthur Main Road, Whitefield', 'KR Puram', 'Koramangala 2nd Block', 'Koramangala 3rd Block', 'Koramangala', 'Hosur Road', 'Rajajinagar', 'Banaswadi', 'North Bangalore', 'Nagawara', 'Hennur', 'Kalyan Nagar', 'New BEL Road', 'Jakkur', 'Rammurthy Nagar', 'Thippasandra', 'Kaggadasapura', 'Hebbal', 'Kengeri', 'Sankey Road', 'Sadashiv Nagar', 'Basaveshwara Nagar', 'Yeshwantpur', 'West Bangalore', 'Magadi Road', 'Yelahanka', 'Sahakara Nagar', 'Peenya'])
        restaurant_type = st.selectbox('Restaurant Type', ['Casual Dining','Quick Bites','Cafe',
            'Delivery',
            'Mess',
            'Dessert Parlor',
            'Pub',
            'Bakery',
            'Takeaway, Delivery',
            'Fine Dining',
            'Beverage Shop',
            'Sweet Shop',
            'Bar',
            'Confectionery',
            'Kiosk',
            'Food Truck',
            'Microbrewery, Casual Dining',
            'Lounge',
            'Bar, Casual Dining',
            'Food Court',
            'Cafe, Bakery',
            'Dhaba',
            'Microbrewery',
            'Pub, Bar',
            'Lounge, Bar',
            'Food Court, Dessert Parlor',
            'Casual Dining, Sweet Shop',
            'Food Court, Casual Dining',
            'Casual Dining, Microbrewery',
            'Sweet Shop, Dessert Parlor',
            'Bakery, Beverage Shop',
            'Lounge, Casual Dining',
            'Cafe, Food Court',
            'Beverage Shop, Cafe',
            'Cafe, Dessert Parlor',
            'Dessert Parlor, Bakery',
            'Microbrewery, Pub',
            'Club',
            'Casual Dining, Irani Cafee',
            'Fine Dining, Lounge',
            'Bar, Quick Bites',
            'Bakery, Kiosk',
            'Pub, Microbrewery',
            'Microbrewery, Lounge',
            'Fine Dining, Microbrewery',
            'Fine Dining, Bar',
            'Mess, Quick Bites',
            'Dessert Parlor, Kiosk',
            'Bhojanalya',
            'Casual Dining, Quick Bites',
            'Pop Up',
            'Cafe, Bar',
            'Casual Dining, Lounge',
            'Bakery, Sweet Shop',
            'Microbrewery, Bar',
            'Cafe, Lounge',
            'Bar, Pub',
            'Lounge, Cafe',
            'Club, Casual Dining',
            'Quick Bites, Mess',
            'Quick Bites, Meat Shop',
            'Quick Bites, Kiosk',
            'Lounge, Microbrewery',
            'Food Court, Beverage Shop',
            'Dessert Parlor, Food Court'
        ]
    )
        cuisine_types = st.selectbox('Cuisine Types', ['North Indian, Mughlai, Chinese', 'Chinese, North Indian, Thai',
           'Cafe, Mexican, Italian',
           'North Indian, Street Food, Biryani', 'Chinese, Mughlai',
           'North Indian, Chinese, Arabian, Momos'])
        cost_for_two = st.number_input('Cost for Two', min_value=40, max_value=6000)
        restaurant_service = st.selectbox('Restaurant Service', ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
           'Drinks & nightlife', 'Pubs and bars'])
        
        # Log information about user input
        logging.info("User input collected.")

        # Create a dictionary from user input
        user_input = {
            'online_orders': online_orders,
            'book_table': book_table,
            'total_votes': total_votes,
            'restaurant_location': restaurant_location,
            'restaurant_type': restaurant_type,
            'cuisine_types': cuisine_types,
            'cost_for_two': cost_for_two,
            'restaurant_service': restaurant_service
        }
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([user_input])
        
        # Make prediction
        prediction = predict_rating(input_df)

        # Display prediction
        st.subheader('Predicted Restaurant Rating:')
        st.success(f"The predicted rating for the restaurant is: {prediction[0]:.2f}")

        # Log successful prediction
        logging.info("Prediction successful.")

        # # Additional Features Section
        # st.markdown("## Additional Features")
        
        # Add images, descriptions, or other features here

        # Footer
        st.markdown(
            """
            ---
            ⚡️ Developed by Suyash Kumbhar | 🌟 
            """
        )

    except Exception as e:
        # Log any errors that occur
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

# Log information about the end of the application
logging.info("Streamlit app completed.")

if __name__ == '__main__':
    main()
