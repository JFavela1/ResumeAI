"""One-off analysis: Mateo resume vs Adobe ML intern posting.

Usage:
  cp .env.example .env
  # edit .env and set OPENAI_API_KEY=sk-...

  python scripts/run_mateo_adobe_analysis.py
"""
import json
import os
import sys
from pathlib import Path

# Repo root on path when run as: python scripts/run_mateo_adobe_analysis.py
_root = Path(__file__).resolve().parents[1]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from resumeai.config import bootstrap_env

bootstrap_env()

from resumeai import analyze

RESUME = """Mateo Boccalato
Phone: (305) 724-2908 | Email: mateoboccalato@gmail.com | LinkedIn/GitHub: Mateo-Boccalato
OBJECTIVE
Aspiring AI/ML Engineer skilled in developing full-stack solutions that combine machine learning models with modern web technologies to
deliver real-world impact.
EDUCATION
San Diego State University San Diego, CA
Bachelor of Science in Computer Science Expected Graduation: May 2027
TECHNICAL SKILLS
Programming Languages: Python, Java, JavaScript
Developmental tools: VS Code, Jupyter Notebook, IntelliJ IDEA, Cursor
Libraries: Scikit-Learn, PyTorch, Pandas
Software and Utilities: GitHub, Vercel, Appwrite, Word, Excel
Languages: English, Portuguese, Spanish
WORK/PROJECT EXPERIENCE
RelaTech Labs/RevelOnward | AI/ML developer Intern | Sept 2025 – Dec 2025
▪
Fine-tuned custom GPT models for high school students with ADHD
▪
Built and optimized RAG pipelines by collecting personalized academic material to enhance model relevance and accuracy.
▪
Collaborated with users to turn learning difficulties into technical model specifications, improving study engagement.
Spanglish Movies | Lead Developer June 2025 - August 2025
▪
Engineered a cross-platform transactional video on demand(TVOD) infrastructure for Floutv.
▪
Integrated Shopify for payments and secure DRM delivery.
▪
Developed the QR-based routing, enabling premium movie rentals across Roku, SamsungTV , ClaroTV , and Plex.
Movie Revenue Prediction Model "FilmBrain" | Python | React JS Feb 2024 - May 2024
▪
Trained a CatBoost Regression model with advanced feature engineering to deliver real-time box-office accurate predictions.
▪
Created a movie database containing 6000+ movies with 12 data elements using TMDB and IMDB's API's.
▪
Achieved an RMSE = $25M and an MAE = $19M.
▪
Built a simple web application using React for the front end, integrating the model using a Flask API
AI Club | Technical Lead | Python | Prompt Engineering Sept 2022 – Mar 2023
▪
Led and managed a team in developing an X-ray-based bone fracture detection model.
▪
Communicated technical concepts effectively to diverse audiences
▪
Met weekly to convene progress and explore new topics
▪
Resolved software and hardware issues efficiently to maintain productivity
RELEV ANT COURSEWORK
Data Structures and Algorithms | Java | San Diego State University Aug 2024 - Dec 2024
▪
Applied data structures and algorithms to solve real-world computational problems efficiently
▪
Evaluated trade-offs between time and space complexity when designing scalable solutions
▪
Implemented efficient searching, sorting, and graph-based algorithms to improve performance
▪
Strengthened problem-solving skills through algorithm design and optimization exercises
Introduction to AI | San Diego State University Aug 2025 - Dec 2025
▪
Implemented supervised and unsupervised learning algorithms, including linear regression,
SVMs, k-means, and hierarchical clustering
▪
Applied heuristic and uninformed search methods to problem-solving and agent-based systems
▪
Analyzed neural network architectures and training dynamics for pattern recognition tasks
▪
Explored AI applications in image classification, object tracking, language models, and speech generation
Advanced Programming Languages | San Diego State University Jan 2026 - May 2026
▪
Evaluated programming languages based on industry adoption, community support, and job market relevance
▪
Analyzed best-use scenarios and limitations of different languages for solving practical software problems
▪
Developed the ability to articulate trade-offs between languages for system design and implementation"""

JOB = """Our Company
Changing the world through digital experiences is what Adobe's all about. We give everyone—from emerging artists to global brands— everything they need to design and deliver exceptional digital experiences. We're passionate about empowering people to create beautiful and powerful images, videos, and apps, and transform how companies interact with customers across every screen.

We're on a mission to hire the very best and are committed to creating exceptional employee experiences where everyone is respected and has access to equal opportunity. We realize that new ideas can come from everywhere in the organization, and we know the next big idea could be yours.

The Opportunity
Adobe is looking for a Machine Learning intern who will apply AI and machine learning techniques to big-data problems to help Adobe better understand, lead and optimize the experience of its customers.

By using predictive models, experimental design methods, and optimization techniques, the candidate will be working on the research and development of exciting projects like real-time online media optimization, sales operation analytics, customer churn scoring and management, customer understanding, product recommendation and customer lifetime value prediction.

All 2026 Adobe interns will be co-located hybrid. This means that interns will work between their assigned office and home. Interns will be based in the office where their manager and/or team are located, where they will get the most support to ensure collaboration and the best employee experience. Managers and their organization will determine the frequency they need to go into the office to meet priorities.

What You'll Do

Develop predictive models on large-scale datasets to address various business problems with statistical modeling, machine learning, and analytics techniques.

Develop and implement scalable, efficient, and interpretable modeling algorithms that can work with large-scale data in production systems

Collaborate with product management and engineering groups to develop new products and features.

What You Need to Succeed

Currently enrolled full time and pursuing a Bachelors, Master's or PhD degree in Computer Science, Computer Engineering; or equivalent experience required with an expected graduation date of December 2026 – June 2027

Good understanding of statistical modeling, machine learning, deep learning, or data analytics concepts.

Proficient in one or more programming languages such as Python, Java and C

Familiar with one or more machine learning or statistical modeling tools such as R, Matlab and scikit learn

Strong analytical and quantitative problem-solving ability.

Excellent communication, relationship skills and a team player

Ability to participate in a full-time internship between May-September

About Adobe

Adobe empowers everyone to create through innovative platforms and tools that unleash creativity, productivity and personalized customer experiences. Adobe's industry-leading offerings including Adobe Acrobat Studio, Adobe Express, Adobe Firefly, Creative Cloud, Adobe Experience Platform, Adobe Experience Manager, and GenStudio enable people and businesses to turn ideas into impact, powered by AI and driven by human ingenuity.

Our 30,000+ employees worldwide are creating the future and raising the bar as we drive the next decade of growth. We're on a mission to hire the very best and believe in creating a company culture where all employees are empowered to make an impact. At Adobe, we believe that great ideas can come from anywhere in the organization. The next big idea could be yours.


Let's Adobe together

At Adobe, we believe in creating a company culture where all employees are empowered to make an impact. Learn more about Adobe life, including our values and culture, focus on people, purpose and community, Adobe for All, comprehensive benefits programs, the stories we tell, the customers we serve, and how you can help us advance our mission of empowering everyone to create.

Adobe is proud to be an Equal Employment Opportunity employer. We do not discriminate based on gender, race or color, ethnicity or national origin, age, disability, religion, sexual orientation, gender identity or expression, veteran status, or any other protected characteristic. Learn more.

Adobe aims to make our Careers website and recruiting process accessible to any and all users. If you have a disability or special need that requires accommodation to navigate our website or complete the application process, email accommodations@adobe.com or call +1 408-536-3015.

AI Use Guidelines for Interviews:
Our interviews are designed to reflect your own skills and thinking. The use of AI or recording tools during live interviews is not permitted unless explicitly invited by the interviewer or approved in advance as part of a reasonable accommodation. If these tools are used inappropriately or in a way that misrepresents your work, your application may not move forward in the process.

At Adobe, we empower employees to innovate with AI — and we look for candidates eager to do the same. As part of the hiring experience, we provide clear guidance on where AI is encouraged during the process and where it's restricted during live interviews. See how we think about AI in the hiring experience.

Expected Pay Range:

Our compensation reflects the cost of labor across several U.S. geographic markets, and we pay differently based on those defined markets. The U.S. pay range for this position is $45.00 -- $55.00 hourly. Your recruiter can share more about the specific pay rate for your job location during the hiring process.
State-Specific Notices:

California:

Fair Chance Ordinances

Adobe will consider qualified applicants with arrest or conviction records for employment in accordance with state and local laws and "fair chance" ordinances.

Colorado:

Application Window Notice

If this role is open to hiring in Colorado (as listed on the job posting), the application window will remain open until at least the date and time stated above in Pacific Time, in compliance with Colorado pay transparency regulations. If this role does not have Colorado listed as a hiring location, no specific application window applies, and the posting may close at any time based on hiring needs.

Massachusetts:

Massachusetts Legal Notice

It is unlawful in Massachusetts to require or administer a lie detector test as a condition of employment or continued employment. An employer who violates this law shall be subject to criminal penalties and civil liability."""


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Missing OPENAI_API_KEY. Set it in the environment or in "
            f"{_root / '.env'} (see .env.example).",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Running AgentMatch analysis (Mateo vs Adobe ML intern)...", file=sys.stderr)
    result = analyze(RESUME, JOB, verbose=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
