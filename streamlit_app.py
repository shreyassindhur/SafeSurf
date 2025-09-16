import streamlit as st
import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
import whois
import ssl
import socket
from datetime import datetime
from urllib.parse import urlparse
import os
from bs4 import BeautifulSoup
import re
from streamlit_echarts import st_echarts

# Page config
st.set_page_config(
    page_title="SafeSurf - Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler (you'll need to upload these to your repo)
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model("models/phishing_tf_model.keras")
        scaler_mean = np.load("models/phishing_tf_scaler.npy")
        scaler_scale = np.load("models/phishing_tf_scaler_scale.npy")
        return model, scaler_mean, scaler_scale
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Feature extraction functions
def url_features(url):
    """Extract URL-based features"""
    features = {}
    
    # Basic URL features
    features['url_length'] = len(url)
    features['num_tokens'] = len(url.split('/'))
    features['has_ip'] = bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))
    
    # Calculate entropy
    chars = list(url)
    char_counts = {}
    for char in chars:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    entropy = 0
    for count in char_counts.values():
        p = count / len(chars)
        if p > 0:
            entropy -= p * np.log2(p)
    features['entropy'] = entropy
    
    # Suspicious patterns
    parsed = urlparse(url)
    domain = parsed.hostname or ""
    
    # Free hosting detection
    free_hosts = ['weebly.com', 'wixsite.com', 'webflow.io', 'mystrikingly.com', 
                  'github.io', 'blogspot.com', 'wordpress.com', 'squarespace.com']
    features['is_free_host'] = int(any(host in domain for host in free_hosts))
    
    # Suspicious subdomain
    subdomains = domain.split('.')
    features['has_suspicious_subdomain'] = int(len(subdomains) > 3)
    
    # Link shortener detection
    shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'buff.ly']
    features['is_link_shortener'] = int(any(shortener in domain for shortener in shorteners))
    
    return features

def html_features(html):
    """Extract HTML-based features"""
    features = {}
    
    if not html:
        features.update({
            'has_forms': 0,
            'external_links': 0,
            'num_scripts': 0,
            'hidden_fields': 0
        })
        return features
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Form detection
        forms = soup.find_all('form')
        features['has_forms'] = int(len(forms) > 0)
        
        # External links
        links = soup.find_all('a', href=True)
        external_count = 0
        for link in links:
            if link['href'].startswith('http') and 'javascript:' not in link['href']:
                external_count += 1
        features['external_links'] = external_count
        
        # Scripts
        scripts = soup.find_all('script')
        features['num_scripts'] = len(scripts)
        
        # Hidden fields
        hidden_inputs = soup.find_all('input', type='hidden')
        features['hidden_fields'] = len(hidden_inputs)
        
    except Exception:
        features.update({
            'has_forms': 0,
            'external_links': 0,
            'num_scripts': 0,
            'hidden_fields': 0
        })
    
    return features

def host_features(whois_info, ssl_info):
    """Extract host-based features"""
    features = {}
    features['domain_age_days'] = whois_info.get('domain_age_days', 0)
    features['ssl_valid'] = ssl_info.get('ssl_valid', 0)
    return features

def get_whois_info(domain):
    """Get WHOIS information for domain"""
    try:
        w = whois.whois(domain)
        age_days = 0
        if w.creation_date:
            creation = w.creation_date
            if isinstance(creation, list):
                creation = min([d for d in creation if d is not None])
            if creation:
                age_days = (datetime.now() - creation).days
        
        return {
            'domain_age_days': age_days,
            'registrar': str(w.registrar) if w.registrar else '',
            'creation_date': str(creation) if creation else ''
        }
    except Exception:
        return {'domain_age_days': 0, 'registrar': '', 'creation_date': ''}

def get_ssl_info(domain):
    """Get SSL certificate information"""
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                return {'ssl_valid': 1, 'issuer': str(cert.get('issuer', ''))}
    except Exception:
        return {'ssl_valid': 0, 'issuer': ''}

def predict_phishing(url, model, scaler_mean, scaler_scale):
    """Main prediction function"""
    if not model:
        return "Error: Model not loaded", 0.0, {}
    
    # Get HTML content
    html = ""
    try:
        response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        html = response.text
    except Exception:
        pass
    
    # Extract domain
    parsed = urlparse(url)
    domain = parsed.hostname or ""
    
    # Get features
    url_feats = url_features(url)
    html_feats = html_features(html)
    whois_info = get_whois_info(domain)
    ssl_info = get_ssl_info(domain)
    host_feats = host_features(whois_info, ssl_info)
    
    # Combine all features
    all_features = {**url_feats, **html_feats, **host_feats}
    
    # Ensure all required features are present
    required_features = [
        'url_length', 'num_tokens', 'has_ip', 'entropy', 'domain_age_days', 
        'ssl_valid', 'has_forms', 'external_links', 'num_scripts', 
        'hidden_fields', 'is_free_host', 'has_suspicious_subdomain', 'is_link_shortener'
    ]
    
    for feat in required_features:
        if feat not in all_features:
            all_features[feat] = 0
    
    # Prepare model input
    feature_vector = [all_features[feat] for feat in required_features]
    feature_vector = np.array(feature_vector)
    feature_vector = (feature_vector - scaler_mean) / scaler_scale
    feature_vector = feature_vector.reshape(1, -1)
    
    # Predict
    prob = float(model.predict(feature_vector)[0][0])
    label = "🚨 PHISHING" if prob > 0.5 else "✅ SAFE"
    
    return label, prob, all_features

# Main app
def main():
    # Header
    st.title("🛡️ SafeSurf - Phishing Detector")
    st.markdown("### Protect yourself from malicious websites")
    
    # Load model
    model, scaler_mean, scaler_scale = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("About SafeSurf")
        st.write("SafeSurf uses advanced machine learning to detect phishing websites and protect you from online threats.")
        
        st.header("How it works")
        st.write("1. Enter a URL")
        st.write("2. We analyze multiple factors")
        st.write("3. Get instant results")
        
        st.header("Team")
        st.write("Shreyas Sindhur")
        st.write("Sujal Phapale")
        st.write("Manasvi Singh")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Check a URL")
        url_input = st.text_input(
            "Enter the website URL you want to check:",
            placeholder="https://example.com",
            help="Enter the complete URL including http:// or https://"
        )
        
        check_button = st.button("🔍 Check URL", type="primary", use_container_width=True)
        
        if check_button and url_input:
            if not url_input.startswith(('http://', 'https://')):
                st.error("Please enter a valid URL starting with http:// or https://")
            else:
                with st.spinner("Analyzing URL... This may take a few seconds."):
                    try:
                        label, probability, features = predict_phishing(url_input, model, scaler_mean, scaler_scale)
                        
                        # Results
                        st.header("🔍 Analysis Results")
                        
                        # Main result
                        if "PHISHING" in label:
                            st.error(f"**{label}** - Probability: {probability:.2%}")
                            st.warning("⚠️ This website may be dangerous. Do not enter personal information.")
                        else:
                            st.success(f"**{label}** - Probability: {(1-probability):.2%}")
                            st.info("✅ This website appears to be legitimate.")

                        # Stylish Odometer (Gauge)
                        st.markdown("#### Security Level")
                        # Always show 0 = phishing (red), 100 = safe (green)
                        security_score = int((1 - probability) * 100)
                        gauge_option = {
                            "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
                            "series": [
                                {
                                    "name": "Security Level",
                                    "type": "gauge",
                                    "min": 0,
                                    "max": 100,
                                    "splitNumber": 5,
                                    "axisLine": {
                                        "lineStyle": {
                                            "color": [
                                                [0.5, "#d9534f"],   # Red for 0-50 (phishing)
                                                [0.8, "#f0ad4e"],   # Orange for 50-80 (warning)
                                                [1, "#5cb85c"]      # Green for 80-100 (safe)
                                            ],
                                            "width": 30
                                        }
                                    },
                                    "pointer": {"width": 6},
                                    "detail": {"formatter": "{value}%", "fontSize": 24},
                                    "data": [
                                        {"value": security_score, "name": "Security"}
                                    ]
                                }
                            ]
                        }
                        st_echarts(options=gauge_option, height="300px")

                        # Feature details
                        with st.expander("📊 Detailed Analysis"):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.subheader("URL Features")
                                st.write(f"URL Length: {features.get('url_length', 0)}")
                                st.write(f"Number of Tokens: {features.get('num_tokens', 0)}")
                                st.write(f"Has IP Address: {'Yes' if features.get('has_ip', 0) else 'No'}")
                                st.write(f"URL Entropy: {features.get('entropy', 0):.2f}")
                                
                            with col_b:
                                st.subheader("Security Features")
                                st.write(f"Domain Age (days): {features.get('domain_age_days', 0)}")
                                st.write(f"SSL Valid: {'Yes' if features.get('ssl_valid', 0) else 'No'}")
                                st.write(f"Has Forms: {'Yes' if features.get('has_forms', 0) else 'No'}")
                                st.write(f"External Links: {features.get('external_links', 0)}")
                        
                    except Exception as e:
                        st.error(f"Error analyzing URL: {str(e)}")
        
        elif check_button:
            st.warning("Please enter a URL to check.")
    
    with col2:
        st.header("🛡️ Stay Safe Online")
        st.info("**Tips to avoid phishing:**")
        st.write("• Check URLs carefully")
        st.write("• Look for HTTPS")
        st.write("• Verify sender identity")
        st.write("• Don't click suspicious links")
        st.write("• Use SafeSurf to check!")
        
        st.markdown("[Learn more about phishing](https://www.phishing.org/what-is-phishing)")

    # Footer
    st.markdown("---")
    st.markdown("🛡️ **SafeSurf** - Your trusted phishing protection | Made with ❤️ using Streamlit")

if __name__ == "__main__":
    main()