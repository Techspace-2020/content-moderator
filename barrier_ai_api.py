import os
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper import NewspaperTools
from agno.tools.website import WebsiteTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.wikipedia import WikipediaTools
import google.generativeai as genai

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

import hashlib
import diskcache
import json  # Ensure JSON module is imported
import re

# Load environment variables from a .env file
load_dotenv()

# Retrieve the API key from the environment variable
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY is not set in the environment variables")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Create the Fake News Classification agent
fake_news_agent = Agent(
    # model=Gemini(id="gemini-2.0-flash"),  # 15 RPM, 1500 req/day
    # model=Gemini(id="gemini-2.5-pro-exp-03-25"),  # 5 RPM, 25 req/day
    model=Gemini(id="gemini-2.5-flash-preview-04-17"),  # 10 RPM, 500 req/day
    tools=[
        DuckDuckGoTools(search=True, news=True),
        NewspaperTools(),
        WikipediaTools(),
        YFinanceTools(enable_all=True),
        WebsiteTools(),
    ],
    structured_outputs=True,
    markdown=True,
    name="BARRIER - protects from fake or scam content",
    instructions="""
        # 🧠 BARRIER AI - Fake News, scam, Profanity, finincial fraud, harmful/toxic speech etc. Detection Agent

        ## 🛠️ Use below tools based on context:
        - `DuckDuckGo`: for real-time web & news search (top 10 or more results)
        - `NewspaperTools`: for parsing and summarizing articles
        - `WikipediaTools`: for factual lookup and historical data
        - `YFinanceTools`: for financial data and stock-related verification
        - `WebsiteTools`: for reading and extracting raw content from URLs
        
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
                - `Needs Review`: If the trust score is between 40% and 50%, indicating that further verification is required.
                - `Highly Unlikely`: If the trust score is below 30%, suggesting that the content is very likely to be false.
                - `Verified`: If the content has been cross-checked with multiple reliable sources and confirmed to be true.
                - `Disputed`: If the content has conflicting information from different sources, requiring careful consideration.

            15. **Verify Financial statements**
                - To handle all financial-related queries, use YFinanceTools exclusively. This tool is designed to provide up-to-date and accurate information on stock prices, market data, financial metrics, and expert recommendations. Whether you're looking for real-time stock prices, historical performance, or market trends, YFinanceTools is the go-to resource. For stock analysis, it also offers buy, hold, or sell recommendations from analysts, along with key metrics like P/E ratios, dividend yields, and earnings reports. Simply input your query related to any financial data—be it about stocks, market performance, or recommendations—and let YFinanceTools deliver the most reliable information available. For all financial-related matters, YFinanceTools ensures that you get precise and current data, making it your primary tool for any financial verification.

            16. Flag Unsafe Content
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

                    **Risk Levels:**
                        - **0** - No Risk: Content is safe and does not fall into any unsafe categories.
                        - **1** - Low Risk: Content may be harmful but not inherently dangerous.
                        - **2** - Medium Risk: Content falls into categories that may be harmful but does not require immediate moderation.
                        - **3** - High Risk: Content is harmful or dangerous and requires immediate action.

                17. Types of Fake news/flase information:
                    1. Clickbait
                        These are stories that are deliberately fabricated to gain more website visitors and increase advertising revenue for websites. Clickbait stories use sensationalist headlines to grab attention and drive click-throughs to the publisher website, normally at the expense of truth or accuracy.
                    2. Propaganda
                        Stories that are created to deliberately mislead audiences, promote a biased point of view or particular political cause or agenda.
                    3. Satire/Parody
                        Lots of websites and social media accounts publish fake news stories for entertainment and parody. For example; The Onion, Waterford Whispers, The Daily Mash, etc.
                    4. Sloppy Journalism
                        Sometimes reporters or journalists may publish a story with unreliable information or without checking all of the facts which can mislead audiences. For example, during the U.S. elections, fashion retailer Urban Outfitters published an Election Day Guide, the guide contained incorrect information telling voters that they needed a ‘voter registration card’. This is not required by any state in the U.S. for voting.
                    5. Misleading Headings
                        Stories that are not completely false can be distorted using misleading or sensationalist headlines. These types of news can spread quickly on social media sites where only headlines and small snippets of the full article are displayed on audience newsfeeds.
                    6. Biased/Slanted News
                        Many people are drawn to news or stories that confirm their own beliefs or biases and fake news can prey on these biases. Social media news feeds tend to display news and articles that they think we will like based on our personalised searches.
                    7. Imposter Content
                        When genuine sources are impersonated with false, made-up sources. This is dangerous as it relates to information with no factual basis being presented in the style of a credible news source or article to make it look like a legitimate source.
                    8. Manipulated Content
                        When real information or imagery is manipulated to deceive, as with a doctored photo or video. This can be used to mislead people or create a false narrative about something or someone. 

                    **How to spot False Information?**
                    1.Who is sharing the story?
                        Check if the the social media account sharing the post is verified. Most public figures and media outlets display a “blue badge or check mark” which means the account has been authenticated. This can mean the content of the post is more likely to be reliable, although not always. 
                    2. Take a closer look
                        Check the source of the story, do you recognise the website? Is it a credible/reliable source? If you are unfamiliar with the site, look in the about section or find out more information about the author.
                    3. Look beyond the headline 
                        Check the entire article, many fake news stories use sensationalist or shocking headlines to grab attention. Often the headlines of fake new stories are in all caps and use exclamation points.
                    4. Check other sources
                        Are other reputable news/media outlets reporting on the story? Are there any sources in the story? If so, check they are reliable or if they even exist!
                    5. Check the facts
                        Stories with false information often contain incorrect dates or altered timelines. It is also a good idea to check when the article was published, is it current or an old news story?
                    6. Check your biases
                        Are your own views or beliefs affecting your judgement of a news feature or report?
                    7. Is it a joke?
                        Satirical sites are popular online and sometimes it is not always clear whether a story is just a joke or parody… Check the website, is it known for satire or creating funny stories?

                    **Fact checking sites**
                    Snopes: snopes.com/
                    PolitiFact: politifact.com
                    Fact Check: factcheck.org/
                    BBC Reality Check: bbc.com/news/reality-check
                    Channel 4 Fact Check: channel4.com/news/factcheck
                    Reverse image search from Google: google.com/reverse-image-search
            
                
    """,
    expected_output="""
                {   
                    "classification": "<Likely True | Unverified | Likely Fake | Needs Review | Highly Unlikely | Verified | Disputed>",
                    "score": "<numerical trust score from 0% to 100%>",
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
    goal="The agents main goal is to analyze provided statements, articles, or claims to determine their factual accuracy based on verifiable information. It should also assess the content for any harmful or unsafe categories, assigning an appropriate risk level. Based on this analysis, BARRIER AI classifies the content as Likely True, Unverified, or Likely Fake, and assigns a numerical trust score reflecting the degree of confidence in the classification.",
    debug_mode=True,
    add_name_to_instructions=True,
)

# Create a new agent for input classification
input_classifier_agent = Agent(
    add_name_to_instructions=True,
    debug_mode=True,
    # model=Gemini(id="gemini-2.0-flash"),  # Lightweight model for faster classification
    model=Gemini(
        id="gemini-2.0-flash-lite"
    ),  # Even lighter model for basic classification tasks 30 RPM, 1500 req/day
    markdown=True,
    name="InputClassifier",
    instructions="""
        You are an input classification agent. Your task is to analyze the input message and classify it into one of the following categories:
        - Claim/News/Article/Statement: If the input is a factual statement, news, or claim that requires fact-checking or it spreads hate,violent,harm,bias,profanity,scam, etc.
        - General/Chit-Chat: If the input is a casual conversation or unrelated to fact-checking.
    """,
    expected_output="""
        {
            "classification": "<Claim/News/Article/Statement | General/Chit-Chat>"
        }
        """,
    description="Classifies input messages to determine if they require fact-checking or are general chit-chat.",
    goal="Classify input messages into relevant categories for further processing.",
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

# Set CORS policy: allow only trusted frontend origins (adjust as needed)
ALLOWED_ORIGINS = [
    "http://localhost:8000",  # Local frontend (React, etc.)
    "http://127.0.0.1:8000",
    "http://localhost:5000",  # chat room url
    "chrome-extension://bgkimgppeklbopjofjaaihhhocgnappk",
    # Add your production frontend domain(s) here:
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],  # Only allow necessary methods
    allow_headers=["Authorization", "Content-Type"],  # Restrict to needed headers
)

# Initialize diskcache (persistent, thread/process safe)
CACHE_DIR = "./barrier_cache"
CACHE_EXPIRY_SECONDS = 60 * 60 * 24  # 24 hours
cache = diskcache.Cache(CACHE_DIR)


def make_cache_key(prefix: str, message: str) -> str:
    """Generate a secure cache key using a hash of the normalized message."""
    normalized = message.strip().lower()
    msg_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{prefix}:{msg_hash}"


# Optimize DiskCache usage with memoize decorator
@cache.memoize(expire=CACHE_EXPIRY_SECONDS)
def classify_message(message: str):
    classification_response: RunResponse = input_classifier_agent.run(message=message)
    if isinstance(classification_response.content, str):
        return json.loads(classification_response.content.strip("```json\n"))
    return classification_response.content


@cache.memoize(expire=CACHE_EXPIRY_SECONDS)
def factcheck_message(message: str):
    response: RunResponse = fake_news_agent.run(message=message)
    if isinstance(response.content, str):
        try:
            return json.loads(response.content.strip("```json\n"))
        except json.JSONDecodeError:
            # If the output is the fallback error message, do not cache it
            if (
                response.content.strip()
                == '{"type": "text", "data": "The response could not be processed. Please try again."}'
            ):
                # Return a special value to indicate no-cache
                raise ValueError("Do not cache this output")
            return response.content.strip()
    return response.content


@app.post("/analyze")
def analyze_message(input: MessageInput):
    try:
        # --- Classification without caching if response is invalid ---
        classification_result = classify_message(input.message)

        # Check the classification result
        if classification_result["classification"] == "General/Chit-Chat":
            response = {
                "type": "text",
                "data": "I am BARRIER AI, designed to fact-check claims. The one you selected looks like Chit-Chat."
            }
            return response

        if classification_result["classification"] == "Claim/News/Article/Statement":
            try:
                factcheck_result = factcheck_message(input.message)
            except ValueError:
                # Return the fallback error message directly, do not cache
                return {
                    "type": "text",
                    "data": "The response could not be processed. Please try again.",
                }
            parsed_result = None

            if isinstance(factcheck_result, str):
                try:
                    # Improved regex to remove markdown code block markers and whitespace
                    cleaned = re.sub(
                        r"^```json\s*|^```\s*|```$",
                        "",
                        factcheck_result.strip(),
                        flags=re.MULTILINE,
                    ).strip()
                    # Extract the first JSON object from the string, even if there is text before/after
                    match = re.search(r"({[\s\S]*})", cleaned)
                    if match:
                        json_str = match.group(1)
                        parsed_result = json.loads(json_str)
                    else:
                        raise json.JSONDecodeError("No JSON object found", cleaned, 0)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}\nCleaned string: {cleaned}")
                    return {
                        "type": "text",
                        "data": "The response could not be processed. Please try again.",
                    }
            elif isinstance(factcheck_result, dict):
                parsed_result = factcheck_result

            if parsed_result:
                # Validate the parsed result to ensure it contains expected keys
                required_keys = [
                    "classification",
                    "score",
                    "top_related_urls",
                    "related_fact",
                    "risk_level",
                    "unsafe_categories",
                    "reason_for_unsafe_classification",
                ]
                if all(key in parsed_result for key in required_keys):
                    response = {
                        "type": "json",
                        "data": {
                            "classification": parsed_result.get(
                                "classification", "Unknown"
                            ),
                            "score": parsed_result.get("score", 0),
                            "top_related_urls": parsed_result.get(
                                "top_related_urls", []
                            ),
                            "related_fact": parsed_result.get("related_fact", ""),
                            "risk_level": parsed_result.get("risk_level", 0),
                            "unsafe_categories": parsed_result.get(
                                "unsafe_categories", []
                            ),
                            "reason_for_unsafe_classification": parsed_result.get(
                                "reason_for_unsafe_classification", ""
                            ),
                        },
                    }
                    return response

            # Avoid caching invalid responses
            return {
                "type": "text",
                "data": "The response could not be processed. Please try again.",
            }

    except Exception as e:
        # Avoid caching exceptions
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app if this script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
