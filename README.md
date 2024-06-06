# Project Media QA

Project Media QA allows you to ask questions about any uploaded/recorded audio, youtube videos powered by Groq's whisper for speech to text transcription API and Groq's LLM API.

## Prerequisites

Before running the project, ensure you have the following:

- Python 3.10 (the project has been tested with this version)
- Groq API key

## Setup

1. Set up your Groq API key as an environment variable:

   ```bash
   export GROQ_API_KEY=<YOUR_API_KEY>
   ```

   Replace `<YOUR_API_KEY>` with your actual Groq API key.

2. Create a virtual environment:

   ```bash
   virtualenv -p python3 venv
   ```

3. Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To run the application locally, use the following command:

```bash
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

This command starts the Streamlit application and makes it accessible at `http://localhost:8080`.

## Usage

1. Launch the application by accessing `http://localhost:8080` in your web browser.

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.

2. Create a new branch for your feature or bug fix.

3. Make your changes and commit them with descriptive commit messages.

4. Push your changes to your forked repository.

5. Submit a pull request to the main repository, explaining your changes and their benefits.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions regarding this project, please feel free to contact the project maintainer at [skapadia@groq.com](mailto:skapadia@groq.com).