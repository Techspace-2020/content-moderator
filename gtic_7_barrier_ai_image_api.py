import os
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper import NewspaperTools
from agno.tools.website import WebsiteTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.wikipedia import WikipediaTools

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

import hashlib
import diskcache
import json  # Ensure JSON module is imported
import re

# At the top of the file, after existing imports
import logging
import json
import re
import hashlib
import diskcache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from agno.media import Image
from agno.agent import RunResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file
load_dotenv()

# Retrieve the API key from the environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")


# Create the Fake News Classification agent
fake_news_agent = Agent(
    model=Gemini(id="gemini-2.5-flash-preview-05-20"),  # 10 RPM, 500 req/day
    tools=[
        DuckDuckGoTools(search=True, news=True, cache_results=True),
        NewspaperTools(),
        WikipediaTools(),
        YFinanceTools(enable_all=True, cache_results=True),
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
                - Based on source credibility, author legitimacy, headline style, fact-check alignment, and sentiment neutrality, assign a trust score ranging from 0% to 100%.

            14. Content Classification (Text-Specific)  
                - **Validated (90–100%)**: Verified by multiple trusted sources; highly reliable.  
                - **Likely Accurate (75–89%)**: Strong evidence supports it; only minor uncertainties.  
                - **Uncertain (50–74%)**: Evidence is mixed or incomplete; needs further checking.  
                - **Review Required (40–49%)**: Clear doubts or inconsistencies; deeper scrutiny needed.  
                - **Conflicted (30–39%)**: Credible sources disagree; treat with caution.  
                - **Likely False (15–29%)**: Strong signs of inaccuracy or fabrication.  
                - **Almost Certainly False (0–14%)**: Virtually no credible support; extremely unlikely to be true.  

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
                Respond ONLY with a valid JSON matching this structure and Do not add any explanation. Return only JSON.

                {   
                    "classification": "<Validated | Likely Accurate | Uncertain | Review Required | Conflicted | Likely False | Almost Certainly False>",
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

# image agent of Barrier AI
fake_image_agent = Agent(
    model=Gemini(id="gemini-2.5-flash-preview-05-20"),  # 10 RPM, 500 req/day
    tools=[
        DuckDuckGoTools(search=True, news=True, cache_results=True),
        NewspaperTools(cache_results=True),
        WikipediaTools(cache_results=True),
        YFinanceTools(enable_all=True, cache_results=True),
        WebsiteTools(cache_results=True),
    ],
    structured_outputs=True,
    markdown=True,
    name="BARRIER – Adaptive Shield Against Fake, Scam & Harmful Visual Content",
    instructions="""
       # 🔐 BARRIER – Image Risk Classification & Fact Verification Agent

           ## 🎯 Mission:
                Proactively safeguard the integrity of visual media through a dual‑system approach—fast, heuristic screening paired with thorough, evidence‑based validation—to detect and neutralize manipulated, misleading, or harmful imagery by continuously cross‑verifying against authoritative, real‑time sources.

            ## 🧰 Supported Tools:
            - `DuckDuckGo`: Reverse lookups and contextual web/news search.
            - `NewspaperTools`: Extract timelines, summaries, and context from news articles.
            - `WikipediaTools`: Verify entities (people, places, events, symbols).
            - `WebsiteTools`: Parse and extract raw content from URLs.
            - `YFinanceTools`: Validate any financial data shown (stock charts, tickers, dashboards).

            ## 🪜 Workflow:

            1. **Image Intake & Pre‑Processing**  
                - **Normalization & Enhancement**  
                    - Auto‑resize and reformat to a standard resolution and aspect ratio for consistent analysis.  
                    - Apply denoising and sharpening filters to improve clarity without introducing artifacts.  
                    - Adjust color balance and contrast to reveal subtle details.  
                - **Structural Analysis**  
                    - Identify and segment key regions (faces, text areas, logos) using object‑detection models.  
                    - Generate image masks for foreground/background separation.  
                - **Alt‑Text Generation**  
                    - Use a vision‑language model to produce a concise, descriptive caption covering scene, objects, people, and actions.  
                    - Include detected attributes (e.g., clothing, expressions, colors).  
                - **OCR & Text Extraction**  
                    - Run high‑precision OCR on all detected text regions (memes, screenshots, captions).  
                    - Capture font style, size, and orientation metadata for context.  
                    - Tag extracted text with spatial coordinates for downstream fact‑checking.  

            2. **Metadata & EXIF Inspection**  
                - (Internal) Extract creation date, device info, geolocation tags, editing history.

            3. **Reverse Search & Origin Trace**  
                - **Query Construction**  
                    - Combine alt‑text keywords, OCR‑extracted text, and visual descriptors into enriched search strings.  
                    - Generate variants with synonyms, translations, and event‑specific terms for broader coverage.  
                - **DuckDuckGo Reverse Image Search**  
                    - Submit the image URL or upload directly to retrieve visually similar results.  
                    - Aggregate the top 10 matches, capturing source URLs, snippet context, and publication timestamps.  
                - **Earliest Appearance Identification**  
                    - Sort results by date to pinpoint the first‑seen instance.  
                    - Cross‑validate publication dates with NewspaperTools and WebsiteTools.  
                - **Contextual Categorization**  
                    - Label each match as “Breaking News,” “Stock Photo Library,” “Social Media Post,” or “Archive/Blog.”  
                    - Highlight mismatches between the claimed context and the original usage.  
                - **Version & Edit Tracking**  
                    - Document variations in resolution, cropping, watermarks, or overlays to map editing history.  
                - **Source Credibility Assessment**  
                    - Evaluate domain reputations (e.g., official news vs. unknown blogs) and note any flagged or low‑trust sites.  

            4. **Contextual & Semantic Verification**  
                - **Claim & Theme Extraction**  
                    - Parse the auto‑generated alt‑text, OCR output, and surrounding captions to identify explicit claims or themes (who, what, when, where).  
                - **Multi‑Source Summarization (NewspaperTools)**  
                    - Pull full articles that reference the image or event.  
                    - Extract key facts: timeline of events, direct quotes, named sources, and any caveats.  
                    - Generate a concise summary with publication dates and author attribution.  
                - **Temporal Alignment**  
                    - Cross‑reference event dates, photo timestamps, and publication times to verify sequence integrity.  
                    - Flag any discrepancies (e.g., image predates claimed event).  
                - **Geographic Corroboration**  
                    - Use WikipediaTools to validate place names, landmarks, and regional context.  
                    - Match GPS metadata (if available) against known locations.  
                - **Entity Validation & Relationship Mapping**  
                    - Identify people, organizations, and objects in the image.  
                    - Confirm existence and role via WikipediaTools and DuckDuckGo (e.g., check if person was present at event).  
                - **Web Context Retrieval (WebsiteTools)**  
                    - Scrape paragraphs, captions, and metadata from pages where the image appears.  
                    - Capture adjacent text for sentiment, intent, and narrative framing.  
                - **Consistency & Contradiction Detection**  
                    - Compare facts from different sources to highlight conflicting accounts.  
                    - Note shifts in tone (objective report vs. opinion piece) or sensational language.  
                - **Source Credibility Scoring**  
                    - Rate each source by domain authority, author reputation, and recency.  
                    - Weight findings accordingly in the final context_score.  

           5. **Image Manipulation & Generation Detection**  
                - **AI‑Generated / Deepfake**  
                    - Check for GAN artifacts: unnatural noise patterns, repeating textures, interpolation glitches.  
                    - Analyze facial landmarks: inconsistent eye spacing, warped mouths, irregular skin pores.  
                    - Evaluate lighting & shadows: mismatched light sources or missing cast shadows.  
                    - Audio–Visual Sync (for video): lip‑sync errors, audio drift against lip movements.  
                    - Upscaling Artifacts: checkerboard patterns from neural upscaling.  
                - **Photoshopped / Edited**  
                    - Clone‑stamp & Copy‑Move Detection: repeated pixel blocks, mirrored regions.  
                    - Edge‑Boundary Analysis: soft or blurred seams where layers were composited.  
                    - Color Inconsistency: mismatched color temperature or saturation between regions.  
                    - Shadow & Reflection Mismatch: incorrect geometry or missing reflections.  
                    - Layer Metadata: check for “Photoshop” or “GIMP” tags in editing history.  
                - **Composite & Splicing**  
                    - Multi‑source Blending: elements from different origins pasted together.  
                    - Perspective Distortion: inconsistent vanishing points or horizon lines.  
                    - Resolution Mismatch: areas of differing sharpness or compression levels.  
                - **Out‑of‑Context Use**  
                    - Temporal Mismatch: image timestamp predates the claimed event.  
                    - Geographic Mismatch: landmarks or signage don’t match claimed location.  
                    - Caption vs. Content Discrepancy: extracted text or headline doesn’t align with scene.  
                - **Breaking‑News Fake**  
                    - Source Traceback: check if first appearance is from non‐news blogs or stock libraries.  
                    - Rapid Viral Spread: image pops up on unverified channels before credible outlets.  
                    - Reverse Search Timestamp: earliest match predates the reported date by days/weeks.  
                - **Synthetic Style & AI Filters**  
                    - Neural Style Artifacts: brush‑stroke patterns or painterly textures not matching original medium.  
                    - HDR / Tone‑Mapping Overdrive: unnatural dynamic range extremes.  
                - **Deep Image Matting & Remapping**  
                    - Background Replacement: jagged edges around foreground subjects.  
                    - Depth‑map Inconsistency: flat or misaligned depth fields in synthetic composites. 

            5.1. **AI Generation Detection**  
                - **GAN Artifact Analysis**: Leverage deep‑learning forensics to scan for repeating noise patterns, checkerboard artifacts, and interpolation glitches common in generative adversarial networks.  
                - **Facial & Structural Anomalies**: Apply landmark consistency checks—eye spacing, jawline symmetry, ear placement—to flag improbable geometries in human subjects or objects.  
                - **Texture & Surface Inconsistency**: Evaluate micro‑texture uniformity: overly smooth skin, glassy reflections, or unrealistic material rendering indicate synthetic origin.  
                - **Lighting & Shadow Coherence**: Cross‑validate light source direction and shadow geometry across all elements; mismatches reveal compositing or AI synthesis.  
                - **Upscaling & Denoising Signatures**: Detect neural upscaling fingerprints—checkerboard grids, denoiser residue—using frequency‑domain analysis.  
                - **Model Watermark Detection**: Identify hidden AI model fingerprints or metadata markers inserted by common generators (e.g., Stable Diffusion, Midjourney).  
                - **Error Level Analysis (ELA)**: Compare compression levels across the image to highlight regions with disparate error rates indicative of synthetic editing.  
                - **Photo‑Response Non‑Uniformity (PRNU) Analysis**: Extract sensor noise patterns and cross‑match against known device fingerprints to detect absence of authentic camera noise.  
                - **Color Palette & Spectrum Anomalies**: Analyze color distributions for unnatural saturation spikes or uniform gradients typical of AI‑generated visuals.  
                - **Semantic Coherence Checks**: Verify logical consistency of scene elements (e.g., floating objects, impossible reflections, anatomical impossibilities).  
                - **Contextual Plausibility Assessment**: Cross‑reference visual content against known event footage or image databases to spot novel, unverifiable scenes.  
                - **Temporal & Motion Consistency** (for video): Examine frame‑by‑frame continuity, optical flow, and motion blur artifacts to detect interpolation or frame‑generation.  
                - **Cross‑Platform Dataset Matching**: Query curated real‑world image repositories to ensure the subject is not a known synthetic template or AI training asset.  

            6. **Embedded Text & Meme Claims**  
                - **With Text**:  
                    - **OCR Extraction**: Use high‑accuracy OCR to capture all embedded words, numbers, and symbols.  
                    - **Linguistic Analysis**: Detect sensationalist language, clickbait phrasing, or manipulated quotes.  
                    - **Entity Recognition**: Identify names, dates, figures, slogans, hashtags, and verify via WikipediaTools or DuckDuckGo.  
                    - **Cross‑Source Verification**: Match extracted text against reputable news sites, official statements, or fact‑check databases.  
                    - **Contextual Consistency**: Ensure the tone and language align with the depicted scene and known event timeline.  
                    - **Statistical Validation**: Check any numeric claims (election results, COVID data, financial figures) against primary data sources or YFinanceTools.  

                - **No Text**:  
                    - **Logo & Branding Identification**: Recognize corporate logos, uniforms, signage, or watermarks and verify origin.  
                    - **Landmark & Environment Analysis**: Match scenery, architecture, vehicle plates, or landscape to known locations via WikipediaTools or web search.  
                    - **Actor & Object Recognition**: Identify public figures, products, or objects in the image and cross‑check their presence at the claimed event.  
                    - **Visual Cue Correlation**: Use contextual details (weather, foliage, crowds) to confirm season, geography, or event specifics.  
                    - **Metadata Tie‑In**: Correlate inferred claims with metadata timestamps and geotags to validate temporal and spatial assertions.  
            
            6.1 Types of Fake news/flase information:
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

            7. **Unsafe Content Classification**  
                - **Hate Imagery**: Symbols, slurs, or actions targeting protected groups.  
                - **Violence / Gore**: Graphic harm, physical assault, weaponry.  
                - **Adult / NSFW**: Nudity, sexual acts, or erotic content.  
                - **Animal Cruelty**: Depictions of abuse or neglect.  
                - **Political Propaganda**: Designed to influence elections or civic processes.  
                - **Child Exploitation**: Any sexualized or abusive depiction of minors.  
                - **Terrorism & Extremist Content**: Promotion or glorification of terrorist acts or groups.  
                - **Self‑Harm / Suicide Promotion**: Content encouraging self‑injury or suicide.  
                - **Medical Misinformation**: False or dangerous health-related imagery (e.g., bogus treatment charts).  
                - **Privacy Violation / Doxing**: Images exposing personal or sensitive data without consent.  
                - **Illicit Behavior Encouragement**: Instructions for theft, hacking, drug manufacture.  
                - **Disinformation / Conspiracy Imagery**: Symbols or visuals tied to baseless conspiracies.  
                - **Graphic Accidents / Disaster Porn**: Gratuitous images of crashes, catastrophes purely for shock.  
                - **Hoarding / Neglect Conditions**: Extreme living conditions that may exploit the subject’s suffering.  

            8. **Risk Level Assignment (0–3)**  
                - **0** Safe: No unsafe elements.  
                - **1** Low Risk: Mildly sensitive (e.g., mild profanity, minimal violence).  
                - **2** Medium Risk: Clear policy violations (hate imagery, misinformation).  
                - **3** High Risk: Severe harm (terrorist content, child exploitation, incitement).

           9. **Final Image Classification & Score**  
                - **Authentic (90–100%)**: Fully corroborated original image with correct context and no manipulation.  
                - **Probably Authentic (75–89%)**: High confidence in genuineness; minor gaps but no significant red flags.  
                - **Undetermined (50–74%)**: Insufficient evidence to confirm or refute authenticity—additional verification needed.  
                - **Review Required (40–49%)**: Noticeable inconsistencies or minor edits detected—expert analysis recommended.  
                - **Conflicting Reports (30–39%)**: Mixed source information or reverse‑search results—credibility contested.  
                - **Probably Manipulated (15–29%)**: Clear signs of editing, AI generation, or context misuse—low originality trust.  
                - **Synthetic/Deepfake (0–14%)**: Near‑certain AI‑generated or maliciously doctored image.  

       
    """,
    expected_output="""
                Respond ONLY with a valid JSON matching this structure and Do not add any explanation. Return only JSON.

                {   
                    "classification": "<Authentic | Probably Authentic | Undetermined | Review Required | Conflicting Reports | Probably Manipulated | Synthetic/Deepfake>",
                    "score": "<numerical trust score from 0% to 100%>",
                    "top_related_urls": [
                        "<URL 1>",
                        "<URL 2>",
                        "<URL 3>"
                    ],
                    "related_fact": "<summary of fact-check result or verified context from trustworthy sources>",
                    "risk_level": "<0 | 1 | 2 | 3>",
                    "unsafe_categories": [
                        "<list of flagged categories, e.g., \"Hate Imagery\", \"Animal Cruelty\", \"Child Exploitation\">"
                    ],
                    "reason_for_unsafe_classification": "<explanation of why the content is flagged as unsafe>"
                }
                """,
    description="BARRIER AI is a vigilant and intelligent agent designed to detect and counter visual misinformation. It analyzes images for signs of manipulation, AI generation, deepfakes, and misleading context. Using a multi-stage pipeline—ranging from OCR and reverse image search to metadata forensics and fact-checking—BARRIER validates embedded claims, traces origin, and assesses semantic alignment. It delivers a trust score and content risk classification, ensuring visual media remains accurate, transparent, and safe across platforms.",
    goal="To rigorously analyze images and visual content for authenticity, contextual accuracy, and potential harm—detecting AI-generated media, deepfakes, hate symbols, manipulated scenes, and fabricated breaking news—to safeguard truth and mitigate visual misinformation across digital platforms.",
    debug_mode=True,
    add_name_to_instructions=True,
)

# Create a new agent for input classification
input_classifier_agent = Agent(
    add_name_to_instructions=True,
    debug_mode=True,
    model=Gemini(id="gemini-2.0-flash-lite"),
    markdown=True,
    name="InputClassifier",
    instructions="""
        You are an input classification agent. Your task is to analyze the input message and classify it into one of the following categories:
        - Claim/News/Article/Statement: If the input is a factual statement, news, or claim that requires fact-checking or it spreads hate,violent,harm,bias,profanity,scam, etc.
        - General/Chit-Chat: If the input is a casual conversation or unrelated to fact-checking.
    """,
    expected_output="""
            Respond ONLY with a valid JSON matching this structure and Do not add any explanation. Return only JSON.

        {
            "classification": "<Claim/News/Article/Statement | General/Chit-Chat>"
        }
        """,
    description="Classifies input messages to determine if they require fact-checking or are general chit-chat.",
    goal="Classify input messages into relevant categories for further processing.",
)


# Define the input and output models for the API
class ImageInput(BaseModel):
    filepath: str | None = None
    url: str | None = None


class MessageInput(BaseModel):
    message: str
    images: list[ImageInput] = []  # Optional list of images (filepath or url)


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
    "http://127.0.0.1:5000",
    "http://13.51.45.209:9000/analyze",
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
async def analyze_message(input: MessageInput):
    try:
        # If images are provided, use fake_image_agent
        if input.images and len(input.images) > 0:
            logger.info(f"Processing request with {len(input.images)} images")
            images = []

            try:
                for img in input.images:
                    if img.filepath:
                        logger.info(f"Processing image from filepath: {img.filepath}")
                        images.append(Image(filepath=img.filepath))
                    elif img.url:
                        logger.info(f"Processing image from URL: {img.url}")
                        images.append(Image(url=img.url))

                if not images:
                    logger.warning("No valid images found in request")
                    return JSONResponse(
                        status_code=400,
                        content={"type": "text", "data": "No valid images provided"},
                    )

                logger.info("Running image agent analysis")
                agent_response: RunResponse = fake_image_agent.run(
                    images=images,
                    message="Analyze the image with given instruction and return the result in JSON format.",
                )
                content = agent_response.content

                # Log the raw response for debugging
                logger.debug(f"Raw agent response: {content}")

                parsed_result = None
                if isinstance(content, str):
                    try:
                        # Strip markdown and extract JSON
                        cleaned = re.sub(
                            r"^```json\\s*|^```\\s*|```$",
                            "",
                            content.strip(),
                            flags=re.MULTILINE,
                        )
                        match = re.search(r"({[\s\S]*})", cleaned)
                        if match:
                            try:
                                parsed_result = json.loads(match.group(1))
                                logger.info("Successfully parsed JSON response")
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"JSON parsing error: {e}\nRaw output: {cleaned}"
                                )
                                return JSONResponse(
                                    status_code=200,
                                    content={
                                        "type": "text",
                                        "data": f"The response could not be processed due to invalid JSON. Raw output: {cleaned}",
                                    },
                                )
                        else:
                            logger.warning("No JSON object found in response")
                            return JSONResponse(
                                status_code=200,
                                content={
                                    "type": "text",
                                    "data": f"No JSON object found in response. Raw output: {cleaned}",
                                },
                            )
                    except Exception as e:
                        logger.error(
                            f"Unexpected error during JSON extraction: {e}\nRaw output: {content}"
                        )
                        return JSONResponse(
                            status_code=200,
                            content={
                                "type": "text",
                                "data": f"Unexpected error during JSON extraction: {e}. Raw output: {content}",
                            },
                        )
                elif isinstance(content, dict):
                    parsed_result = content
                    logger.info("Response was already in dict format")

                if parsed_result:
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
                        response_data = {
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
                        }

                        logger.info(
                            "About to return successful image analysis response"
                        )
                        return JSONResponse(
                            status_code=200,
                            content={"type": "json", "data": response_data},
                        )
                    else:
                        logger.warning(
                            f"Missing required keys in response. Found keys: {list(parsed_result.keys())}"
                        )
                        return JSONResponse(
                            status_code=200,
                            content={
                                "type": "text",
                                "data": f"Missing required keys in response. Found keys: {list(parsed_result.keys())}",
                            },
                        )

                logger.warning("Invalid or incomplete response from image agent")
                return JSONResponse(
                    status_code=200,
                    content={
                        "type": "text",
                        "data": "The response could not be processed. Please try again.",
                    },
                )

            except Exception as e:
                logger.error(f"Error processing images: {str(e)}", exc_info=True)
                logger.error("Exception occurred before sending HTTP response")
                return JSONResponse(
                    status_code=200,
                    content={
                        "type": "text",
                        "data": f"Error processing images: {str(e)}",
                    },
                )

        # --- Handle text-only requests ---
        classification_result = classify_message(input.message)

        # Check the classification result
        if classification_result["classification"] == "General/Chit-Chat":
            response = {
                "type": "text",
                "data": "I am BARRIER AI, designed to fact-check claims. The selected content appears to be casual conversation.",
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
    uvicorn.run(app, host="0.0.0.0", port=6000)
