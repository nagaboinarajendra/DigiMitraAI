from flask import Flask, request, jsonify
import sqlite3
import os
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# Set the path for the SQLite database (persistent storage)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'users_appointment.db')

# Function to initialize the database
def initialize_database():
    # Ensure the directory for the DB exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # If the database doesn't exist, create it
    if not os.path.exists(DB_PATH):
        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.cursor()
            cursor.execute("""CREATE TABLE IF NOT EXISTS users_appointment (
                              id INTEGER PRIMARY KEY AUTOINCREMENT,
                              name TEXT NOT NULL, 
                              mobile_number INTEGER NOT NULL UNIQUE,
                              otp INTEGER NOT NULL, 
                              address TEXT NOT NULL,
                              aadhar_center TEXT NOT NULL,
                              appointment_date TEXT NOT NULL
            )""")
            connection.commit()

# Before the first request, initialize the database
@app.before_first_request
def setup():
    initialize_database()

# Flask API to book an appointment
@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    data = request.json
    name = data.get('name')
    mobile_number = data.get('mobile_number')
    otp = data.get('otp')
    address = data.get('address')
    aadhar_center = data.get('aadhar_center')

    # Validate input
    if not all([name, mobile_number, otp, address, aadhar_center]):
        return jsonify({'error': 'All fields are required'}), 400

    try:
        mobile_number = int(mobile_number)
        otp = int(otp)
    except ValueError:
        return jsonify({'error': 'Mobile number and OTP must be integers'}), 400

    # Check if the mobile number already exists
    try:
        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM users_appointment WHERE mobile_number = ?", (mobile_number,))
            existing_record = cursor.fetchone()
            if existing_record:
                return jsonify({'error': 'Appointment already exists for this mobile number'}), 400

            # Generate a random appointment date (3–7 days from today)
            random_days = random.randint(3, 7)
            appointment_date = (datetime.now() + timedelta(days=random_days)).strftime('%Y-%m-%d')

            # Insert new appointment
            cursor.execute(
                "INSERT INTO users_appointment (name, mobile_number, otp, address, aadhar_center, appointment_date) VALUES (?, ?, ?, ?, ?, ?)",
                (name, mobile_number, otp, address, aadhar_center, appointment_date)
            )
            connection.commit()

        return jsonify({'status': 'Appointment booked successfully', 'appointment_date': appointment_date}), 200
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

# Flask API to fetch appointment status
@app.route('/appointment_status', methods=['GET'])
def appointment_status():
    mobile_number = request.args.get('mobile_number')

    if not mobile_number:
        return jsonify({'error': 'Mobile number is required'}), 400

    try:
        mobile_number = int(mobile_number)
    except ValueError:
        return jsonify({'error': 'Mobile number must be an integer'}), 400

    try:
        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT name, appointment_date FROM users_appointment WHERE mobile_number = ?", (mobile_number,))
            record = cursor.fetchone()

        if record:
            # Send a structured JSON response
            return jsonify({
                'status': 'Appointment found',
                'name': record[0],
                'appointment_date': record[1]
            }), 200
        else:
            return jsonify({'status': 'No appointment found for this mobile number'}), 404
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

if __name__ == "__main__":
    # Initialize the database before running the app
    initialize_database()
    app.run(port=5001, debug=True)