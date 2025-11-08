# DTU Connect 

DTU Connect is an AI-powered platform that helps students of Delhi Technological University (DTU) find peers, societies, and opportunities that match their interests and skills.  
It uses Google Gemini API to generate text embeddings for intelligent matching.

---

##  Features

-  Peer Matcher:  
  Find students with similar skills, interests, or academic backgrounds.

-  Society Recommender: 
  Discover DTU societies that align with your goals and skillset.

-  AI-Powered Matching:  
  Uses Google Gemini’s `text-embedding-004` model to compute similarity scores.

-  Smart Class Bonus: 
  Gives higher match weight to people in the same class section.

-  (Optional upcoming) DTU AI Chat Assistant:  
  Ask DTU-related queries (societies, fests, clubs, or academic help).



 Tech Stack

- Frontend/UI: [Streamlit](https://streamlit.io)  
- AI Model: Google Gemini (`text-embedding-004`)  
- Language: Python  
- Data:`peers.json` and `societies.json`

