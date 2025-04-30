import os
from phi.agent import Agent, RunResponse
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper_tools import NewspaperTools
from phi.tools.website import WebsiteTools
from phi.tools.yfinance import YFinanceTools
from phi.tools.wikipedia import WikipediaTools
import google.generativeai as genai

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel
import uvicorn

# Load environment variables from a .env file
load_dotenv()

# Retrieve the API key from the environment variable
#GOOGLE_API_KEY= "AIzaSyDJaDmmsu3N_nAMx0QjSKC15gido884ekU"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Create the Fake News Classification agent
fake_news_agent = Agent(
    # model=Gemini(id="gemini-2.0-flash"),  # 15 RPM, 1500 req/day
    # model=Gemini(id="gemini-2.5-pro-exp-03-25"),  # 5 RPM, 25 req/day
    model=Gemini(id="gemini-2.5-flash-preview-04-17"),  # 10 RPM, 500 req/day
    tools=[
        DuckDuckGo(search=True, news=True),
        NewspaperTools(),
        WikipediaTools(),
        YFinanceTools(enable_all=True),
        WebsiteTools(),
    ],
    structured_outputs=True,
    prevent_hallucinations=True,
    prevent_tool_hallucinations=True,
    markdown=True,
    name="BARRIER - FakeNewsClassifier",
    instructions="""
        # 🧠 BARRIER AI - Fake News Detection Agent

        ## 🛠️ Use below tools based on context:
        - `DuckDuckGo`: for real-time web & news search (top 10 or more results)
        - `NewspaperTools`: for parsing and summarizing articles
        - `WikipediaTools`: for factual lookup and historical data
        - `YFinanceTools`: for financial data and stock-related verification
        - `WebsiteTools`: for reading and extracting raw content from URLs

        ---
        
        ## 🔍 Step-by-Step Instruction

            1. **Review the Statement**
            - Remove HTML tags, emojis, and special characters.
            - Extract key elements of the content (e.g., Title, Body, Date, Author, Source).

            2. **Evaluate Source Credibility**
                - Search for reviews of the source and confirm whether it's a trusted, reputable outlet.
                - Check if the domain is flagged in reputation databases.

            3. **Verify Author Identity**
                - Investigate the author's legitimacy by searching for their name alongside “journalist” or LinkedIn.
                - Ensure the author's previous works are legitimate and credible.

            4. **Analyze Headline Style**
                - Detect sensational or emotionally charged headlines (e.g., ALL CAPS, clickbait language).

            5. **Extract Key Entities**
                - Use NLP to identify names, places, dates, and events mentioned in the content.

            6. **Cross-Verify Events**
                - Cross-check event-related claims using trusted news outlets (e.g., Reuters, BBC).

            7. **Verify Date of Publication**
                - Ensure the article is recent and relevant, and compare the publication date with the event timeline.

            8. **Verify Media Content**
                - For images: Perform a reverse image search using tools like TinEye or Google Images to detect misuse.
                - For videos: Extract keyframes and search for matching content using video forensics tools (e.g., InVID).

            9. **Fact-Check the Claim**
                - Use fact-checking databases (e.g., Snopes, AltNews) to verify the statement and check for known hoaxes.

            10. **Use Built-in Tools**
                - Utilize tools like WikipediaTools, YFinanceTools, and NewspaperTools for further content validation.

            11. **Run Sentiment Analysis**
                - Flag content that uses manipulative or biased language designed to provoke strong emotional responses.

            12. **Cross-Verify on Social Media**
                - Check platforms like Twitter and Reddit to see if the claim has been debunked or flagged by other users.

            13. **Assign Trust Score**
                - Based on source credibility, author legitimacy, headline style, fact-check alignment, and sentiment neutrality, assign a trust score ranging from 0 to 100.

            14. **Classify Content**
                - `Likely True`: If the trust score is above 80%.
                - `Unverified`: If the trust score is between 50% and 80%.
                - `Likely Fake`: If the trust score is below 50%.

            15. Flag Unsafe Content
                Objective: 
                Identify and categorize content that violates safety policies or legal standards.
                Process: 
                Analyze content against the comprehensive categories below. Assign an appropriate risk level (0-3) based on the severity, explicitness, intent, and potential harm of the identified content. A score of 0 indicates no identified unsafe content.

                Comprehensive Unsafe Content Categories:
                I. Severe Harm & Illegality (Often Highest Risk - Level 3)
                    Child Sexual Abuse Material (CSAM) / Child Exploitation:
                        Depiction or promotion of sexual acts involving minors.
                        Grooming activities.
                        Content facilitating or encouraging child exploitation.
                    Terrorism & Violent Extremism:
                        Promotion or glorification of terrorist organizations or acts.
                        Recruitment for terrorist/extremist groups.
                        Instructions for making weapons or carrying out attacks.
                    Incitement to extremist violence.
                        Non-Consensual Sexual Content (NCII / Revenge Porn):
                        Sharing private, intimate images/videos without consent of depicted individuals.
                        Threats to share such content.
                    Promotion/Glorification of Serious Violent Crimes:
                        Inciting, glorifying, or facilitating severe violence against individuals or groups (e.g., murder, kidnapping, torture, armed robbery).
                        Graphic depictions of extreme violence intended to shock or promote harm.
                    Human Trafficking & Modern Slavery:
                        Content facilitating or promoting the trafficking of persons for labor or sexual exploitation.
                    Illegal & Dangerous Goods/Services Trade:
                        Facilitation of sales/trade of illegal firearms, explosives, illicit drugs, human organs, endangered species products.
                II. Harm to Individuals & Groups (Risk Level 1-3 depending on severity/intent)
                    Hate Speech & Discrimination:
                        Attacks, threats, or demeaning content targeting individuals or groups based on protected characteristics (race, ethnicity, religion, gender, sexual orientation, disability, etc.).
                    Promotion of discriminatory ideologies.
                        Harassment & Bullying:
                    Targeted abuse, threats, intimidation, or humiliation of private individuals.
                        Cyberbullying, stalking.
                        Coordinated harassment campaigns.
                    Self-Harm & Suicide Promotion:
                        Content that promotes, glorifies, or provides instructions for suicide or self-injury.
                        Dangerous weight-loss methods, eating disorder promotion.
                        Note: Sensitive discussions about mental health for support purposes may be permissible, requiring careful context analysis.
                    Graphic Violence & Disturbing Content:
                        Excessively graphic depictions of accidents, death, severe injury, medical procedures, or gore without sufficient context (e.g., news reporting, educational).
                        Intentionally shocking or gratuitously violent imagery.
                    Privacy Violations:
                        Doxing (sharing private/identifying information without consent).
                        Sharing non-public personal data (financial, medical).
                        Non-consensual surveillance or recording in private spaces.
                    Animal Cruelty:
                        Depiction or promotion of gratuitous violence, abuse, or neglect towards animals.

                III. Misinformation, Deception & Fraud (Risk Level 1-3 depending on potential harm)
                    Dangerous Misinformation / Disinformation:
                        Content likely to cause real-world harm (e.g., medical misinformation during a health crisis, false information intended to incite violence or obstruct civic processes).
                        Deepfakes used maliciously.
                    Harmful Conspiracy Theories:
                        Promotion of baseless theories known to incite hatred, violence, harassment, or harmful actions (e.g., targeting groups, undermining public safety).
                    Scams, Fraud & Financial Deception:
                        Phishing attempts, investment scams (e.g., pump-and-dump), fraudulent schemes, impersonation for financial gain.
                    Malicious Impersonation:
                        Impersonating individuals, brands, or organizations to deceive, defraud, or harass (distinct from parody/satire).

                IV. Potentially Illegal or Regulated Activities (Risk Level 1-2 depending on context/jurisdiction)
                    Promotion of Non-Violent Crimes:
                        Encouraging or providing instructions for theft, vandalism, trespassing, hacking (non-critical infrastructure), etc.
                    Intellectual Property Violations:
                        Unauthorized distribution of copyrighted material (piracy).
                        Counterfeit goods promotion/sale.
                        Trademark infringement.
                    Regulated Goods & Services (Platform/Jurisdiction Dependent):
                        Promotion/sale of regulated items like gambling, non-illicit drugs (e.g., prescription, cannabis where restricted), alcohol, tobacco, certain weapons (where legal but platform-restricted).
                    Defamation / Libel / Slander:
                        False statements presented as fact that harm the reputation of an individual or entity (often legally complex, may require specific complaint).

                V. Explicit & Mature Content (Risk Level 1-2 depending on explicitness/context)
                    Explicit Sexual Content (Pornography):
                        Depiction of explicit sexual acts involving consenting adults.
                        Note: Often requires age-gating or restricted access rather than outright removal, depending on platform policy.
                    Sexual Solicitation:
                        Offering or seeking commercial sexual services.

                VI. Platform Integrity & Security (Risk Level 1-3 depending on impact)
                    Spam:
                        Unsolicited, repetitive, or commercially deceptive content.
                    Malware / Malicious Software:
                        Distribution of viruses, spyware, or links to harmful websites.
                        Promotion of hacking tools/services targeting critical infrastructure or causing widespread harm.
                    Coordinated Inauthentic Behavior:
                        Using fake accounts or networks to artificially amplify content, manipulate discussions, or mislead users about origin/intent.

                Risk Level Assignment (0-3):
                    0: Safe: No identified unsafe content.
                    1: Low Risk / Sensitive: Content may border on violating policy, be low-impact spam, contain mature themes requiring labeling, or represent minor infractions (e.g., minor IP issue).
                    2: Medium Risk / Harmful: Clear violation of policy with potential for harm (e.g., harassment, scams, significant misinformation, non-graphic hate speech, regulated goods violations).
                    3: High Risk / Severe Harm: Content is illegal or poses an immediate or severe threat (e.g., CSAM, terrorism, incitement to serious violence, NCII, dangerous misinformation with high impact potential).

                Important Considerations:
                    Context is Crucial: 
                        News reporting, educational content, artistic expression, or condemnation of harmful acts often requires different handling than promotion or glorification.
                    Intent Matters: 
                        Was the content intended to harm, deceive, incite, or exploit?
                    Severity & Scale: 
                        How explicit, graphic, or widespread is the content? What is the potential real-world impact?
                    Regional & Legal Differences: 
                        Laws and cultural norms vary, potentially affecting categorization and risk level.
                    Platform Policies: 
                        Specific platform rules will dictate the exact definitions and actions taken for each category.

                **Risk Levels:**
                - **0** - No Risk: Content is safe and does not fall into any unsafe categories.
                - **1** - Low Risk: Content may be harmful but not inherently dangerous.
                - **2** - Medium Risk: Content falls into categories that may be harmful but does not require immediate moderation.
                - **3** - High Risk: Content is harmful or dangerous and requires immediate action.

                16. Generate Structured Output
                    The result should be returned **ONLY** as a single, valid JSON object adhering strictly to the following structure. Do not include any text before or after the JSON object (like Here is the JSON output:
                    {
                    "classification": "<Likely True | Unverified | Likely Fake>",
                    "score": "<numerical trust score from 0 to 100>",
                    "top_related_urls": [
                        "<URL 1>",
                        "<URL 2>",
                        "<URL 3>"
                    ],
                    "related_fact": "<summary of fact-check result or verified context from trustworthy sources>",
                    "risk_level": "<0 | 1 | 2 | 3>",
                    "unsafe_categories": [
                        "<list of flagged categories, e.g., \"Hate Speech\", \"Misinformation\">"
                    ],
                    "reason_for_unsafe_classification": "<explanation of why the content is flagged as unsafe>"
                    }

    """,
    description="BARRIER AI is an intelligent, autonomous agent designed to fact-check claims, analyze news articles, and assess the credibility of social media content in real-time. Using advanced natural language processing (NLP) and integrated fact-checking tools, BARRIER AI verifies content against reliable sources and identifies harmful or unsafe categories. The agent autonomously determines whether a statement is true, false, or unverified, and assigns a risk level based on potential harm. It ensures transparency by providing evidence-based reasoning and detailed outputs.",
    task="The agents main goal is to analyze provided statements, articles, or claims to determine their factual accuracy based on verifiable information. It should also assess the content for any harmful or unsafe categories, assigning an appropriate risk level. Based on this analysis, BARRIER AI classifies the content as Likely True, Unverified, or Likely Fake, and assigns a numerical trust score reflecting the degree of confidence in the classification.",
    debug_mode=True,
)

# Create a new agent for input classification
input_classifier_agent = Agent(
    debug_mode=True,
    # model=Gemini(id="gemini-2.0-flash"),  # Lightweight model for faster classification
    model=Gemini(
        id="gemini-2.0-flash-lite"
    ),  # Even lighter model for basic classification tasks 30 RPM, 1500 req/day
    structured_outputs=True,
    prevent_hallucinations=True,
    prevent_tool_hallucinations=True,
    markdown=True,
    name="InputClassifier",
    instructions="""
        You are an input classification agent. Your task is to analyze the input message and classify it into one of the following categories:
        - Claim/News/Article/Statement: If the input is a factual statement, news, or claim that requires fact-checking.
        - General/Chit-Chat: If the input is a casual conversation or unrelated to fact-checking.

        Respond with a JSON object in the following format:
        {
            "classification": "<Claim/News/Article/Statement | General/Chit-Chat>"
        }
    """,
    description="Classifies input messages to determine if they require fact-checking or are general chit-chat.",
    task="Classify input messages into relevant categories for further processing.",
)


# Define the input and output models for the API
class MessageInput(BaseModel):
    message: str


class ResponseOutput(BaseModel):
    classification: str
    score: int
    top_related_urls: list[str]
    related_fact: str
    risk_level: int
    unsafe_categories: list[str]
    reason_for_unsafe_classification: str


# Initialize FastAPI app
app = FastAPI()

handler = Mangum(app)

# Replace the Hugging Face classifier with Gemini for claim/news detection
@app.post("/analyze")
def analyze_message(input: MessageInput):
    try:
        # Step 1: Classify the input message
        classification_response: RunResponse = input_classifier_agent.run(
            message=input.message
        )

        # Parse the classification response
        if isinstance(classification_response.content, str):
            import json

            classification_result = json.loads(
                classification_response.content.strip("```json\n")
            )
        else:
            classification_result = classification_response.content

        # Check the classification result
        if classification_result["classification"] == "General/Chit-Chat":
            return {
                "type": "text",
                "data": "I am BARRIER AI, designed to fact-check claims, analyze news articles, and assess the credibility of social media content in real-time.",
            }

        # Step 2: If classified as Claim/News/Article/Statement, process with the fake_news_agent
        response: RunResponse = fake_news_agent.run(message=input.message)

        # Parse and return the response from fake_news_agent
        if isinstance(response.content, str):
            try:
                response_content = json.loads(response.content.strip("```json\n"))
                return {"type": "json", "data": response_content}
            except json.JSONDecodeError:
                return {"type": "text", "data": response.content.strip()}
        else:
            return {"type": "json", "data": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app if this script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9091)
