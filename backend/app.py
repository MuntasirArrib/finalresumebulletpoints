from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
import boto3
import os

app = Flask(__name__)
CORS(app)

# Set AWS credentials
os.environ["AWS_PROFILE"] = "ArribIAM"

# Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

modelID = "anthropic.claude-v2"

# Initialize Bedrock LLM
llm = Bedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_tokens_to_sample": 1000, "temperature": 0.9}
)

def generate_bullet_points(category, role, job_description, resume, years_of_experience):
    prompt = PromptTemplate(
        input_variables=["category", "role", "job_description", "resume", "years_of_experience"],
        template=(
            f"You are a Career Coach specialized in helping individuals tailor their resumes to specific job roles and categories. "
            f"Consider the following details:\n\n"
            f"1. Category: '{category}'\n"
            f"2. Role: '{role}'\n"
            f"3. User's Current Resume: '{resume}'\n"
            f"4. Job Description: '{job_description}'\n"
            f"5. User's Years of Experience: {years_of_experience} years\n\n"
            f"Using this information, generate exactly 3 highly relevant, STAR-formatted bullet points that:\n"
            f"- Clearly quantify the user's achievements.\n"
            f"- Align closely with the job description.\n"
            f"- Highlight the user's key skills and experience that make them a strong fit for the role.\n\n"
            f"Ensure each bullet point is impactful and tailored to the role '{role}' in the '{category}' category."
        )
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)

    response = bedrock_chain({'category': category, 'role': role, 'job_description': job_description, 'resume': resume, 'years_of_experience': years_of_experience})
    bullet_points = response['text'].split('\n')
    bullet_points = [point.strip() for point in bullet_points if point.strip().startswith('-')]
    return bullet_points[:3]

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    category = data.get('category')
    role = data.get('role')
    job_description = data.get('job_description')
    resume = data.get('resume')
    years_of_experience = data.get('years_of_experience')
    if not category or not role or not job_description or not resume or not years_of_experience:
        return jsonify({'error': 'Missing required parameters'}), 400
    bullet_points = generate_bullet_points(category, role, job_description, resume, years_of_experience)
    return jsonify({'bullet_points': bullet_points})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
