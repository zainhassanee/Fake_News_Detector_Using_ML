import gradio as gr
import joblib

# Load the saved model
model = joblib.load("model.pkl")

# Define prediction function
def classify_news(text):
    if not text.strip():
        return "Please enter a news article."
    
    prediction = model.predict([text])[0]
    return "🟢 Real News" if prediction == 1 else "🔴 Fake News"

# Define the Gradio interface
interface = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(
        lines=10,
        label="Enter News Article",
        placeholder="Type or paste a news article here..."
    ),
    outputs=gr.Text(label="Prediction"),
    title="📰 Fake News Detector",
    description="This app uses a machine learning model trained on real and fake news data to classify whether an article is real or fake.",
    theme="default"
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
