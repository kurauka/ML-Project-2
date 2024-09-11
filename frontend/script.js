document.getElementById('file-input').addEventListener('change', function () {
    document.getElementById('predict-btn').disabled = false;
});

document.getElementById('predict-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('https://<YOUR_BACKEND_URL>/predict', {
        method: 'POST',
        body: formData,
    });
    
    const result = await response.json();
    document.getElementById('result').innerText = `Class: ${result.class}, Confidence: ${result.confidence}`;
});
