import streamlit as st
import spacy
import feedparser
from urllib.parse import quote
import time
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

US_NEWS_OUTLETS = {
    "NPR": "npr.org",
    "Associated Press": "apnews.com",
    "Reuters": "reuters.com",
    "CNN": "cnn.com",
    "NBC News": "nbcnews.com",
    "ABC News": "abcnews.go.com",
    "CBS News": "cbsnews.com",
    "USA Today": "usatoday.com",
    "The New York Times": "nytimes.com",
    "Washington Post": "washingtonpost.com",
    "Fox News": "foxnews.com",
    "MSNBC": "msnbc.com",
    "Bloomberg": "bloomberg.com",
    "The Wall Street Journal": "wsj.com",
    "The Hill": "thehill.com",
    "Politico": "politico.com",
    "Axios": "axios.com",
    "Vox": "vox.com",
    "Time": "time.com",
    "Newsweek": "newsweek.com",
    "National Review": "nationalreview.com",
    "The Atlantic": "theatlantic.com",
    "Chicago Tribune": "chicagotribune.com",
    "Houston Chronicle": "houstonchronicle.com",
    "LA Times": "latimes.com",
    "New York Daily News": "nydailynews.com",
    "The Denver Post": "denverpost.com",
    "Miami Herald": "miamiherald.com",
    "The Seattle Times": "seattletimes.com",
    "The Boston Globe": "bostonglobe.com",
    "San Francisco Chronicle": "sfchronicle.com",
    "The Philadelphia Inquirer": "inquirer.com",
    "The Atlanta Journal-Constitution": "ajc.com",
    "The Oregonian": "oregonlive.com",
    "The Dallas Morning News": "dallasnews.com",
    "The Detroit News": "detroitnews.com",
    "The Star Tribune": "startribune.com",
    "The Arizona Republic": "azcentral.com",
    "The Tampa Bay Times": "tampabay.com",
    "The Charlotte Observer": "charlotteobserver.com",
    "The Kansas City Star": "kansascity.com",
    "The Sacramento Bee": "sacbee.com",
    "St. Louis Post-Dispatch": "stltoday.com",
    "The San Diego Union-Tribune": "sandiegouniontribune.com",
    "The Baltimore Sun": "baltimoresun.com",
    "Pittsburgh Post-Gazette": "post-gazette.com",
    "The Plain Dealer": "cleveland.com",
    "Milwaukee Journal Sentinel": "jsonline.com",
    "The Indianapolis Star": "indystar.com",
    "ProPublica": "propublica.org",
    "The Intercept": "theintercept.com",
    "HuffPost": "huffpost.com",
    "BuzzFeed News": "buzzfeednews.com",
    "Slate": "slate.com",
    "Salon": "salon.com",
    "Daily Beast": "thedailybeast.com",
    "Reason": "reason.com",
    "Mother Jones": "motherjones.com",
    "Breitbart": "breitbart.com"
}

@st.cache_data(ttl=3600)
def get_google_news_posts(domain):
    base_url = "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en"
    query = f"site:{domain}"
    url = f"{base_url}&q={quote(query)}"
    feed = feedparser.parse(url)
    opinion_keywords = {"opinion", "editorial", "commentary", "op-ed", "analysis", "perspective", "viewpoint"}
    return [(entry.get("title", ""), entry.get("link", ""), domain) 
            for entry in feed.entries 
            if entry.get("title") and entry.get("link") 
            and not any(keyword.lower() in entry.get("title", "").lower() for keyword in opinion_keywords)]

def fetch_all_us_posts(selected_outlets, progress_bar, status_text):
    posts = []
    total_sources = len(selected_outlets)
    
    for i, outlet in enumerate(selected_outlets):
        domain = US_NEWS_OUTLETS[outlet]
        status_text.text(f"Fetching posts from {outlet} ({i+1}/{total_sources})...")
        outlet_posts = get_google_news_posts(domain)
        posts.extend(outlet_posts)
        progress_bar.progress((i + 1) / total_sources)
        time.sleep(0.5)
    
    return posts

def cluster_posts(posts, similarity_threshold, status_text=None):
    if status_text:
        status_text.text("Clustering articles...")
    
    if not posts:
        return {}, {}
    
    clusters = defaultdict(list)
    docs = list(nlp.pipe([post[0] for post in posts], disable=["parser"]))
    phrase_to_doc = {}
    
    for i, doc in enumerate(docs):
        key_phrase = None
        for ent in doc.ents:
            if len(ent.text.split()) >= 2 and ent.label_ in ["PERSON", "ORG", "GPE", "EVENT"]:
                key_phrase = ent.text
                break
        if not key_phrase:
            words = [token.text.lower() for token in doc if not token.is_stop and token.pos_ in ["NOUN", "PROPN", "VERB"]]
            key_phrase = " ".join(words[:2]) if len(words) >= 2 else words[0] if words else posts[i][0].split()[0].lower()
        
        clusters[key_phrase].append(posts[i])
        phrase_to_doc[key_phrase] = doc
    
    merged_clusters = {}
    cluster_similarities = {}
    used_phrases = set()
    
    for phrase, post_list in clusters.items():
        if phrase in used_phrases:
            continue
        
        merged_clusters[phrase] = post_list
        used_phrases.add(phrase)
        
        if len(post_list) > 1:
            sim_scores = []
            cluster_docs = [phrase_to_doc[phrase]] + [nlp(post[0], disable=["parser"]) for post in post_list[1:]]
            for i in range(len(cluster_docs)):
                for j in range(i + 1, len(cluster_docs)):
                    sim_scores.append(cluster_docs[i].similarity(cluster_docs[j]))
            cluster_similarities[phrase] = sum(sim_scores) / len(sim_scores) if sim_scores else 0
        else:
            cluster_similarities[phrase] = 0
        
        for other_phrase, other_posts in clusters.items():
            if other_phrase in used_phrases or phrase == other_phrase:
                continue
            if phrase_to_doc[phrase].similarity(phrase_to_doc[other_phrase]) > similarity_threshold:
                merged_clusters[phrase].extend(other_posts)
                used_phrases.add(other_phrase)
                if other_posts:
                    sim_scores = [phrase_to_doc[phrase].similarity(nlp(post[0], disable=["parser"])) for post in other_posts]
                    cluster_similarities[phrase] = (cluster_similarities[phrase] + sum(sim_scores)) / (len(sim_scores) + 1)
    
    return merged_clusters, cluster_similarities

def classify_clusters(clusters, similarities, selected_outlets):
    underreported = {}
    overreported = {}
    
    for topic, post_list in clusters.items():
        num_articles = len(post_list)
        unique_outlets = len(set(post[2] for post in post_list))
        
        if num_articles <= 3 and unique_outlets <= 3:
            reason = f"{num_articles} article{'s' if num_articles > 1 else ''} from {unique_outlets} outlet{'s' if unique_outlets > 1 else ''}"
            underreported[topic] = (post_list, reason)
        
        elif num_articles > 7 and unique_outlets >= 4:
            reason = f"{num_articles} articles from {unique_outlets} outlets"
            overreported[topic] = (post_list, reason)
    
    return underreported, overreported

def clean_text(text):
    outlet_names = list(US_NEWS_OUTLETS.keys())
    cleaned_text = text.strip()
    for outlet in outlet_names:
        cleaned_text = cleaned_text.replace(outlet, "").replace(f"{outlet} -", "").replace(f"- {outlet}", "")
    return " ".join(cleaned_text.split()).strip("- ")

st.set_page_config(page_title="Newscoop: Uncover America's Under-reported and Over-reported Stories", layout="wide")

# Sidebar for parameters only
with st.sidebar:
    st.header("Parameters")
    
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.7,
        max_value=0.95,
        value=0.85,
        step=0.05,
        help="Controls how similar articles must be to form a cluster. Higher values (e.g., 0.95) mean stricter clustering, grouping only very similar stories, while lower values (e.g., 0.7) allow looser, broader clusters."
    )
    
    overreported_min_articles = st.slider(
        "Minimum Articles for Overreported Stories",
        min_value=5,
        max_value=20,
        value=8,
        step=1,
        help="Sets the minimum number of articles required to classify a story as overreported. Increase this (e.g., 10 or more) to focus on stories with heavy coverage, or decrease it (e.g., 5) to include stories with moderate attention."
    )
    
    overreported_min_outlets = st.slider(
        "Minimum Outlets for Overreported Stories",
        min_value=2,
        max_value=10,
        value=4,
        step=1,
        help="Defines the minimum number of news outlets that must cover a story for it to be considered overreported. Higher values (e.g., 8) highlight stories with widespread outlet coverage, while lower values (e.g., 2) include stories with fewer sources."
    )
    
    st.markdown('<br />', unsafe_allow_html=True)
    st.markdown('<a href="https://3okash.github.io/home/blog/morningscoop.html" target="_blank">READ ME</a>', unsafe_allow_html=True)

# Main content area
st.title("Newscoop 🕵️")
st.markdown('<h3 style="font-size: 30px;">Uncover America\'s Under-reported and Over-reported Stories</h3>', unsafe_allow_html=True)

# Moved outlet selection and button to content area
all_outlets = list(US_NEWS_OUTLETS.keys())
top_10_outlets = [
    "NPR", "Associated Press", "Reuters", "The New York Times", "Washington Post",
    "CNN", "Fox News", "MSNBC", "Bloomberg", "The Wall Street Journal"
]
selected_outlets = st.multiselect(
    "Select News Outlets (10+ recommended for best results)",
    all_outlets,
    default=top_10_outlets
)

if st.button("Dig in 🕵️"):
    if not selected_outlets:
        st.warning("Please select at least one news outlet.")
    elif len(selected_outlets) < 10:
        st.warning("Selecting fewer than 10 outlets may limit results.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        with st.spinner("Fetching from Google News ..."):
            posts = fetch_all_us_posts(selected_outlets, progress_bar, status_text)
            if not posts:
                st.warning("No recent posts found.")
            else:
                clusters, similarities = cluster_posts(posts, similarity_threshold, status_text)
                if not clusters:
                    st.warning("Couldn’t cluster posts.")
                else:
                    st.session_state["posts"] = posts
                    st.session_state["clusters"] = clusters
                    st.session_state["similarities"] = similarities
                    st.session_state["selected_outlets"] = selected_outlets
        progress_bar.empty()
        status_text.empty()

st.markdown('<hr>', unsafe_allow_html=True)

if "posts" in st.session_state and "clusters" in st.session_state:
    posts = st.session_state["posts"]
    clusters = st.session_state["clusters"]
    similarities = st.session_state["similarities"]
    selected_outlets = st.session_state["selected_outlets"]
    
    st.success(f"Analyzed {len(posts)} articles from {len(selected_outlets)} sources.")
    
    underreported_clusters, overreported_clusters = classify_clusters(clusters, similarities, selected_outlets)
    
    if underreported_clusters:
        st.subheader("Under-reported Stories")
        for topic, (post_list, reason) in list(underreported_clusters.items())[:7]:
            for text, link, _ in post_list:
                st.write(f"- {clean_text(text)} ({reason})")
    else:
        st.info("No underreported stories found.")
    
    if overreported_clusters:
        st.subheader("Over-reported Stories")
        for topic, (post_list, reason) in list(overreported_clusters.items())[:10]:
            representative_text, representative_link, _ = post_list[0]
            st.write(f"- {clean_text(representative_text)} ({reason})")
    else:
        st.info("No over-reported stories found.")