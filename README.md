# Potato Disease Detection using FastAPI and TensorFlow

This repository contains a machine learning application for detecting diseases in potato leaves. The project is built using **FastAPI** for the backend and serves a TensorFlow-based model that classifies potato leaves as either:
- **Early Blight**
- **Late Blight**
- **Healthy**

A web-based frontend is included, allowing users to upload images or use their webcam for real-time predictions.

---

## Table of Contents
- [Technologies Used](#technologies-used)
- [Project Features](#project-features)
- [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Frontend Instructions](#frontend-instructions)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Technologies Used

### Backend:
- [FastAPI](https://fastapi.tiangolo.com/) - High-performance backend API framework
- [TensorFlow](https://www.tensorflow.org/) - Machine learning library used for model inference
- [gdown](https://pypi.org/project/gdown/) - Google Drive downloader to fetch the pre-trained model
- [Uvicorn](https://www.uvicorn.org/) - ASGI web server to run the FastAPI app

### Frontend:
- **HTML/CSS**
- [TailwindCSS](https://tailwindcss.com/) - Utility-first CSS framework
- **JavaScript** - Used for handling file uploads and webcam integration

---

## Project Features
- **Potato Leaf Disease Detection**: Classifies potato leaves into Early Blight, Late Blight, or Healthy.
- **REST API**: Provides endpoints for health checks and image prediction.
- **Frontend**: Built with HTML/CSS and JavaScript, enabling users to upload images or capture them using a webcam.
- **Model Loading**: Automatically downloads the machine learning model from Google Drive during the server startup.

---

## Installation

### Clone the repository:

```bash
git clone https://github.com/your-username/potato-disease-detection.git
cd potato-disease-detection
```

### Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the application:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5500
```

---

## Usage

### Backend

1. **Run the FastAPI server** using the command above. This will start the backend at `http://localhost:5500`.
2. **Access the API docs**: Open [http://localhost:5500/docs](http://localhost:5500/docs) in your browser to interact with the API via Swagger UI.
3. **Make Predictions**: Use the `/predict` endpoint to upload an image and get a disease classification.

### Frontend

1. Open the `index.html` file in your browser.
2. Upload an image or capture a photo using your webcam.
3. Click the "Predict" button to see the prediction results.

---

## Endpoints

### `/ping` [GET]
- **Description**: Health check endpoint to ensure the API is running.
- **Response**:
  ```json
  {
    "message": "Hello, Badi"
  }
  ```

### `/predict` [POST]
- **Description**: Predicts the class of the potato leaf image.
- **Request**: Upload an image file.
- **Response**:
  ```json
  {
    "class": "Late Blight",
    "confidence": 0.92
  }
  ```

---

## Frontend Instructions

The web-based frontend allows users to interact with the backend and get disease predictions by uploading images or using their webcam.

- **Image Upload**: Drag and drop an image or click the upload button to select a file from your computer.
- **Webcam Capture**: Use your webcam to capture an image directly for prediction.
- **Prediction**: Once the image is uploaded, click the "Predict" button to receive the classification results.

---

## Future Improvements
- **Deploy on Cloud**: Deploy the backend and frontend on cloud services like **Railway**, **Vercel**, or **Netlify** for broader access.
- **Enhance the Model**: Expand the dataset to include more potato diseases or similar plant diseases.
- **Optimize the Frontend**: Improve user experience and design for better interactivity.

---

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

---

Feel free to fork this project, contribute, and report issues. For any questions, open an issue or contact me!

