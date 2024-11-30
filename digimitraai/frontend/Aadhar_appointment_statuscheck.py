import streamlit as st
import requests

# Function to book an appointment
def book_appointment_ui(name, mobile_number, otp, address, aadhar_center):
    url = "https://aadhar-appointment-stub.onrender.com/book_appointment"
    payload = {
        "name": name,
        "mobile_number": mobile_number,
        "otp": otp,
        "address": address,
        "aadhar_center": aadhar_center
    }
    try:
        response = requests.post(url, json=payload)
        print("Response received:", response.text)  # Log the raw response
        response_data = response.json()  # Attempt to parse the response as JSON

        if response.status_code == 200:
            return f"Appointment successfully booked for {name}. Your appointment is scheduled for {response_data['appointment_date']}."
        else:
            return f"Error: {response_data.get('error', 'Unknown error occurred')}"

    except Exception as e:
        return f"Failed to connect to the server: {str(e)}"

# Function to fetch appointment status
def fetch_status_ui(mobile_number):
    url = f"https://aadhar-appointment-stub.onrender.com/appointment_status?mobile_number={mobile_number}"
    try:
        response = requests.get(url)
        print("Response received:", response.text)  # Log the raw response
        response_data = response.json()  # Attempt to parse the response as JSON

        if response.status_code == 200:
            return f"Hello {response_data['name']}, your appointment is scheduled for {response_data['appointment_date']}."
        else:
            return f"Error: {response_data.get('error', 'No appointment found for this mobile number')}"

    except Exception as e:
        return f"Failed to connect to the server: {str(e)}"

# Streamlit UI
def main():
    st.title("Aadhar Appointment Booking and Status Check")
    
    # Initialize session state variables
    if 'active_button' not in st.session_state:
        st.session_state.active_button = "book"
    
    if 'name' not in st.session_state:
        st.session_state.name = ""
    if 'mobile_number' not in st.session_state:
        st.session_state.mobile_number = ""
    if 'otp' not in st.session_state:
        st.session_state.otp = ""
    if 'address' not in st.session_state:
        st.session_state.address = ""
    if 'aadhar_center' not in st.session_state:
        st.session_state.aadhar_center = ""
    if 'status_number' not in st.session_state:
        st.session_state.status_number = ""
    if 'appointment_confirmed' not in st.session_state:
        st.session_state.appointment_confirmed = False
    if 'appointment_message' not in st.session_state:
        st.session_state.appointment_message = ""
    
    # Set up buttons side by side
    col1, col2 = st.columns(2)
    
    with col1:
        book_button = st.button("Book Appointment", key="book")
    
    with col2:
        status_button = st.button("Check Appointment Status", key="status")
    
    # Reset form if switching to book appointment
    if book_button:
        st.session_state.active_button = "book"
        st.session_state.appointment_confirmed = False  # Reset on switching
        st.session_state.name = ""  # Reset all form fields
        st.session_state.mobile_number = ""
        st.session_state.otp = ""
        st.session_state.address = ""
        st.session_state.aadhar_center = ""
        st.session_state.appointment_message = ""  # Reset any success message
        
    # Reset status number if switching to status check
    elif status_button:
        st.session_state.active_button = "status"
        st.session_state.status_number = ""  # Reset the status number field
    
    # Book Appointment form
    if st.session_state.active_button == "book":
        if st.session_state.appointment_confirmed:
            st.success(st.session_state.appointment_message)
        else:
            st.subheader("Book your Aadhar Appointment")
            
            # User inputs for booking appointment
            name = st.text_input("Name", value=st.session_state.name)
            mobile_number = st.text_input("Mobile Number", value=st.session_state.mobile_number)
            otp = st.text_input("OTP", value=st.session_state.otp)
            address = st.text_area("Address", value=st.session_state.address)
            aadhar_center = st.text_input("Aadhar Center", value=st.session_state.aadhar_center)
            
            if st.button("Book Appointment Now"):
                if not (name and mobile_number and otp and address and aadhar_center):
                    st.error("Please fill all the fields.")
                else:
                    response = book_appointment_ui(name, mobile_number, otp, address, aadhar_center)
                    if "successfully" in response:
                        st.session_state.appointment_confirmed = True
                        st.session_state.appointment_message = response
                        st.experimental_rerun()  # Force a re-run after booking appointment to show the success message immediately
                    else:
                        st.error(response)

    # Check Appointment Status form
    elif st.session_state.active_button == "status":
        st.subheader("Check your Aadhar Appointment Status")
        
        # User input for checking appointment status
        status_number = st.text_input("Enter your Mobile Number to check status", value=st.session_state.status_number)
        
        if st.button("Check Status"):
            if not status_number:
                st.error("Please enter your mobile number.")
            else:
                response = fetch_status_ui(status_number)
                st.write(response)

if __name__ == "__main__":
    main()