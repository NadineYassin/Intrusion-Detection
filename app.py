import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import shap
import tempfile
from collections import defaultdict
try:
    from scapy.all import rdpcap, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Constants and Configuration
# ============================================

FEATURES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size',
    'Subflow Fwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Max',
    'Active Min', 'Idle Mean', 'Idle Max', 'Idle Min'
]

# Simple explanations for each feature (for beginners)
FEATURE_EXPLANATIONS = {
    'Destination Port': 'Which "door" on the server? (80=website, 22=SSH, 443=secure website)',
    'Flow Duration': 'How long did this connection last? (in microseconds)',
    'Total Fwd Packets': 'How many packets did the source send?',
    'Total Length of Fwd Packets': 'Total size of all packets sent by source (bytes)',
    'Fwd Packet Length Max': 'Largest packet sent by source (bytes)',
    'Fwd Packet Length Min': 'Smallest packet sent by source (bytes)',
    'Fwd Packet Length Mean': 'Average packet size sent by source (bytes)',
    'Fwd Packet Length Std': 'How much do packet sizes vary? (standard deviation)',
    'Bwd Packet Length Max': 'Largest packet received back (bytes)',
    'Bwd Packet Length Min': 'Smallest packet received back (bytes)',
    'Bwd Packet Length Mean': 'Average packet size received back (bytes)',
    'Bwd Packet Length Std': 'How much do received packet sizes vary?',
    'Flow Bytes/s': 'Data transfer speed (bytes per second)',
    'Flow Packets/s': 'How many packets per second?',
    'Flow IAT Mean': 'Average time between packets (Inter-Arrival Time)',
    'Flow IAT Std': 'How much does time between packets vary?',
    'Flow IAT Max': 'Longest gap between packets',
    'Flow IAT Min': 'Shortest gap between packets',
    'Fwd IAT Total': 'Total time between all sent packets',
    'Fwd IAT Mean': 'Average time between sent packets',
    'Fwd IAT Std': 'Variation in time between sent packets',
    'Fwd IAT Max': 'Longest gap between sent packets',
    'Fwd IAT Min': 'Shortest gap between sent packets',
    'Bwd IAT Total': 'Total time between all received packets',
    'Bwd IAT Mean': 'Average time between received packets',
    'Bwd IAT Std': 'Variation in time between received packets',
    'Bwd IAT Max': 'Longest gap between received packets',
    'Bwd IAT Min': 'Shortest gap between received packets',
    'Fwd Header Length': 'Size of headers in sent packets',
    'Bwd Header Length': 'Size of headers in received packets',
    'Fwd Packets/s': 'Packets sent per second',
    'Bwd Packets/s': 'Packets received per second',
    'Min Packet Length': 'Smallest packet in the entire flow',
    'Max Packet Length': 'Largest packet in the entire flow',
    'Packet Length Mean': 'Average packet size in the flow',
    'Packet Length Std': 'How much do all packet sizes vary?',
    'Packet Length Variance': 'Another measure of packet size variation',
    'FIN Flag Count': 'How many "connection closing" signals?',
    'PSH Flag Count': 'How many "push data now" signals?',
    'ACK Flag Count': 'How many "message received" confirmations?',
    'Average Packet Size': 'Average size of all packets',
    'Subflow Fwd Bytes': 'Bytes in forward sub-flows',
    'Init_Win_bytes_forward': 'Initial TCP window size (sent)',
    'Init_Win_bytes_backward': 'Initial TCP window size (received)',
    'act_data_pkt_fwd': 'Packets with actual data (not just headers)',
    'min_seg_size_forward': 'Minimum segment size sent',
    'Active Mean': 'Average time connection was active',
    'Active Max': 'Maximum active time',
    'Active Min': 'Minimum active time',
    'Idle Mean': 'Average idle time',
    'Idle Max': 'Maximum idle time',
    'Idle Min': 'Minimum idle time'
}

LABEL_MAPPING = {
    'Normal Traffic': 0,
    'DoS': 1,
    'DDoS': 2,
    'Port Scanning': 3,
    'Brute Force': 4,
    'Web Attacks': 5,
    'Bots': 6
}
REVERSE_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

ATTACK_COLORS = {
    'Normal Traffic': '#28a745',
    'DoS': '#dc3545',
    'DDoS': '#c82333',
    'Port Scanning': '#fd7e14',
    'Brute Force': '#6f42c1',
    'Web Attacks': '#17a2b8',
    'Bots': '#ffc107'
}

ATTACK_DESCRIPTIONS = {
    'Normal Traffic': 'Regular, safe network activity like browsing websites or sending emails.',
    'DoS': 'Denial of Service - Attacker floods a server with so much traffic that it crashes or becomes unavailable to real users.',
    'DDoS': 'Distributed DoS - Same as DoS but the attack comes from thousands of computers at once (often a botnet).',
    'Port Scanning': 'Attacker checks which "doors" (ports) are open on a computer to find vulnerabilities. Like a burglar checking which windows are unlocked.',
    'Brute Force': 'Attacker tries thousands of password combinations until one works. Like trying every possible key on a lock.',
    'Web Attacks': 'Attacks on websites like SQL Injection (tricking the database) or XSS (injecting malicious scripts).',
    'Bots': 'Infected computers controlled remotely by hackers, often used together as a "botnet" for large attacks.'
}

ATTACK_ACTIONS = {
    "DoS": [
        "Enable rate limiting",
        "Block attacking IP",
        "Inspect traffic spikes",
        "Scale server resources"
    ],
    "DDoS": [
        "Activate DDoS mitigation",
        "Block malicious IP ranges",
        "Traffic scrubbing",
        "Notify ISP"
    ],
    "Port Scanning": [
        "Block scanner IP",
        "Close unused ports",
        "Harden firewall rules"
    ],
    "Brute Force": [
        "Lock accounts",
        "Enable MFA",
        "Block attacker IP"
    ],
    "Web Attacks": [
        "Enable Web Application Firewall (WAF)",
        "Patch vulnerabilities",
        "Inspect application logs"
    ],
    "Bots": [
        "Isolate infected host",
        "Run malware scan",
        "Block command-and-control traffic"
    ]
}

# Pre-computed metrics from notebook
METRICS = {
    'XGBoost': {
        'Accuracy': 0.9990,
        'Precision': 0.9991,
        'Recall': 0.9990,
        'F1 Score': 0.9990,
        'Memory (MB)': 1123.05,
        'Training Time (s)': 27.35
    },
    'Random Forest': {
        'Accuracy': 0.9988,
        'Precision': 0.9996,
        'Recall': 0.9988,
        'F1 Score': 0.9992,
        'Memory (MB)': 1614.42,
        'Training Time (s)': 105.98
    }
}

# Pre-computed confusion matrix from XGBoost
CONFUSION_MATRIX = np.array([
    [553, 0, 0, 0, 31, 0, 0],
    [0, 2745, 0, 0, 0, 0, 0],
    [0, 0, 38404, 0, 0, 0, 0],
    [0, 0, 0, 58124, 0, 0, 0],
    [243, 0, 0, 0, 628275, 0, 0],
    [0, 0, 0, 0, 0, 27208, 0],
    [0, 0, 0, 5, 0, 0, 638]
])

CLASS_LABELS = ['Bots', 'Brute Force', 'DDoS', 'DoS', 'Normal Traffic', 'Port Scanning', 'Web Attacks']

PER_CLASS_METRICS = {
    'Bots': {'precision': 0.69, 'recall': 0.95, 'f1': 0.80, 'support': 584},
    'Brute Force': {'precision': 1.00, 'recall': 1.00, 'f1': 1.00, 'support': 2745},
    'DDoS': {'precision': 1.00, 'recall': 1.00, 'f1': 1.00, 'support': 38404},
    'DoS': {'precision': 1.00, 'recall': 1.00, 'f1': 1.00, 'support': 58124},
    'Normal Traffic': {'precision': 1.00, 'recall': 1.00, 'f1': 1.00, 'support': 628518},
    'Port Scanning': {'precision': 0.99, 'recall': 1.00, 'f1': 0.99, 'support': 27208},
    'Web Attacks': {'precision': 0.99, 'recall': 0.99, 'f1': 0.99, 'support': 643}
}

# ============================================
# Load Model and Data
# ============================================

@st.cache_resource
def load_model():
    """Load the trained XGBoost model."""
    model_path = Path(__file__).parent / "models" / "xgboost.joblib"
    return joblib.load(model_path)

@st.cache_resource
def load_explainer():
    """Load the pre-trained SHAP TreeExplainer."""
    explainer_path = Path(__file__).parent / "models" / "shap_explainer.joblib"
    return joblib.load(explainer_path)

# Note: Scaler is NOT needed - the XGBoost model was trained on unscaled data
# The scaler file exists but was only used for SMOTE resampling in the notebook

# ============================================
# Prediction Functions
# ============================================

def predict_single(model, features_df):
    """Make prediction for a single sample.

    Note: The XGBoost model was trained on UNSCALED data in the notebook,
    so we pass features directly without scaling.
    """
    prediction = model.predict(features_df)
    return REVERSE_MAPPING[int(prediction[0])]

def predict_batch(model, features_df):
    """Make predictions for multiple samples.

    Note: The XGBoost model was trained on UNSCALED data in the notebook,
    so we pass features directly without scaling.
    """
    predictions = model.predict(features_df)
    return [REVERSE_MAPPING[int(p)] for p in predictions]

# ============================================
# UI Components
# ============================================

def render_header():
    """Render the app header."""
    st.title("🛡️ Network Intrusion Detection System")
    st.markdown("**AI-powered detection of cyber attacks in network traffic**")
    st.markdown("---")

def render_prediction_result(prediction, actual=None):
    """Render prediction result with styling."""
    if prediction == 'Normal Traffic':
        st.success(f"✅ **Prediction: {prediction}**")
    else:
        st.error(f"🚨 **ATTACK DETECTED: {prediction}**")

    # Show description
    st.info(f"💡 **What is this?** {ATTACK_DESCRIPTIONS.get(prediction, '')}")

    if actual:
        if prediction == actual:
            st.success(f"✓ **Correct!** The actual label was: {actual}")
        else:
            st.warning(f"✗ **Incorrect.** The actual label was: {actual}")

# ============================================
# Tab 1: Live Detection (PCAP + Manual Input)
# ============================================

def load_demo_samples():
    """Load demo samples for the 'Load Example' button."""
    demo_path = Path(__file__).parent / "demo_attacks.csv"
    if demo_path.exists():
        return pd.read_csv(demo_path)
    return None

# ============================================
# Hybrid Detection: Feature Boosting & Rule-Based Override
# ============================================

def detect_attack_behaviors(flows):
    """
    Detect clear attack behaviors at PCAP level for feature boosting.
    
    This function identifies attack patterns that may not be properly
    represented in Python-extracted features due to distribution mismatch
    with CICFlowMeter features.
    
    Uses more aggressive thresholds to catch attacks that might be missed.
    
    Args:
        flows: List of flow dictionaries from extract_flows_from_pcap
    
    Returns:
        List of dictionaries with attack behavior indicators per flow
    """
    behaviors = []
    
    # Calculate global statistics for relative comparison
    all_packet_rates = []
    all_byte_rates = []
    all_syn_ratios = []
    
    for flow in flows:
        if flow['first_seen'] and flow['last_seen']:
            duration = flow['last_seen'] - flow['first_seen']
            if duration > 0:
                packet_rate = len(flow['packets']) / duration
                total_bytes = sum(p['size'] for p in flow['packets'])
                byte_rate = total_bytes / duration
                all_packet_rates.append(packet_rate)
                all_byte_rates.append(byte_rate)
                
                if flow['protocol'] == 6:  # TCP
                    syn_count = flow.get('syn_count', 0)
                    total_packets = len(flow['packets'])
                    if total_packets > 0:
                        syn_ratio = syn_count / total_packets
                        all_syn_ratios.append(syn_ratio)
    
    # Calculate thresholds based on data distribution (more adaptive)
    if all_packet_rates:
        median_packet_rate = np.median(all_packet_rates)
        high_packet_threshold = max(100, median_packet_rate * 2)  # At least 2x median or 100 pps
    else:
        high_packet_threshold = 100
    
    if all_byte_rates:
        median_byte_rate = np.median(all_byte_rates)
        high_byte_threshold = max(1_000_000, median_byte_rate * 2)  # At least 2x median or 1MB/s
    else:
        high_byte_threshold = 1_000_000
    
    for flow in flows:
        behavior = {
            'syn_flood': False,
            'port_scan': False,
            'high_packet_rate': False,
            'high_byte_rate': False,
            'suspicious_pattern': False  # General suspicious pattern
        }
        
        # SYN Flood Detection: More aggressive thresholds
        if flow['protocol'] == 6:  # TCP
            syn_count = flow.get('syn_count', 0)
            ack_count = flow.get('ack_count', 0)
            total_packets = len(flow['packets'])
            
            # SYN flood: Balanced threshold - clear SYN flood pattern
            if syn_count >= 5 and ack_count < syn_count * 0.2 and total_packets >= 10:
                behavior['syn_flood'] = True
            # Also check SYN ratio for very high ratios
            elif total_packets > 0:
                syn_ratio = syn_count / total_packets
                if syn_ratio > 0.3 and syn_count >= 3:  # More than 30% SYN packets
                    behavior['syn_flood'] = True
        
        # Port Scanning Detection: Balanced threshold
        unique_ports = flow.get('unique_dst_ports', 0)
        if unique_ports >= 4:  # Balanced threshold - not too low, not too high
            behavior['port_scan'] = True
        
        # High Packet Rate Detection: Use adaptive threshold
        if flow['first_seen'] and flow['last_seen']:
            duration = flow['last_seen'] - flow['first_seen']
            if duration > 0:
                packet_rate = len(flow['packets']) / duration
                # Use adaptive threshold
                if packet_rate >= high_packet_threshold:
                    behavior['high_packet_rate'] = True
                # Also check if significantly above average
                elif all_packet_rates and packet_rate > np.percentile(all_packet_rates, 90):
                    behavior['high_packet_rate'] = True
        
        # High Byte Rate Detection: Use adaptive threshold
        if flow['first_seen'] and flow['last_seen']:
            duration = flow['last_seen'] - flow['first_seen']
            total_bytes = sum(p['size'] for p in flow['packets'])
            if duration > 0:
                byte_rate = total_bytes / duration
                # Use adaptive threshold
                if byte_rate >= high_byte_threshold:
                    behavior['high_byte_rate'] = True
                # Also check if significantly above average
                elif all_byte_rates and byte_rate > np.percentile(all_byte_rates, 90):
                    behavior['high_byte_rate'] = True
        
        # General suspicious pattern: Only flag very clear suspicious patterns
        # Removed to avoid false positives - let ML model handle ambiguous cases
        
        behaviors.append(behavior)
    
    return behaviors

def boost_features_for_attacks(features_df, attack_behaviors):
    """
    Boost or adjust features when attack behaviors are detected.
    
    This compensates for feature extraction mismatch between Python/scapy
    and CICFlowMeter by artificially adjusting features that indicate attacks.
    
    Uses moderate, targeted boosting only for flows with clear attack behaviors.
    Does NOT apply global boosting to avoid false positives.
    
    Args:
        features_df: DataFrame with extracted features
        attack_behaviors: List of behavior dictionaries from detect_attack_behaviors
    
    Returns:
        DataFrame with boosted features
    """
    boosted_df = features_df.copy()
    
    for idx, behavior in enumerate(attack_behaviors):
        if idx >= len(boosted_df):
            continue
        
        # SYN Flood: Moderate boost for clear SYN flood patterns
        if behavior['syn_flood']:
            # Boost Flow Packets/s and Fwd Packets/s moderately
            boosted_df.loc[idx, 'Flow Packets/s'] *= 3.0
            boosted_df.loc[idx, 'Fwd Packets/s'] *= 3.0
            boosted_df.loc[idx, 'ACK Flag Count'] *= 1.8
            boosted_df.loc[idx, 'Total Fwd Packets'] *= 2.0
        
        # Port Scanning: Moderate boost for clear port scanning
        if behavior['port_scan']:
            # Boost packet rates moderately
            boosted_df.loc[idx, 'Flow Packets/s'] *= 2.5
            boosted_df.loc[idx, 'Fwd Packets/s'] *= 2.5
            boosted_df.loc[idx, 'Total Fwd Packets'] *= 1.8
        
        # High Packet Rate: Moderate boost for high-rate flows
        if behavior['high_packet_rate']:
            boosted_df.loc[idx, 'Flow Packets/s'] *= 2.5
            boosted_df.loc[idx, 'Fwd Packets/s'] *= 2.5
            boosted_df.loc[idx, 'Bwd Packets/s'] *= 1.5
        
        # High Byte Rate: Moderate boost for high-byte-rate flows
        if behavior['high_byte_rate']:
            boosted_df.loc[idx, 'Flow Bytes/s'] *= 3.0
            boosted_df.loc[idx, 'Total Length of Fwd Packets'] *= 2.0
    
    return boosted_df

def rule_based_detection(flows, attack_behaviors):
    """
    Rule-based attack detection to override ML predictions when VERY clear patterns exist.
    
    Priority: Rule-based detection > ML prediction
    
    Uses conservative rules - only overrides when pattern is VERY clear to avoid false positives.
    
    Args:
        flows: List of flow dictionaries
        attack_behaviors: List of behavior dictionaries
    
    Returns:
        List of rule-based predictions (or None if no rule matches)
    """
    rule_predictions = []
    
    for i, (flow, behavior) in enumerate(zip(flows, attack_behaviors)):
        prediction = None
        
        # Only override for VERY clear attack patterns
        
        # SYN Flood -> DoS (only if very clear pattern)
        if behavior['syn_flood']:
            syn_count = flow.get('syn_count', 0)
            total_packets = len(flow['packets'])
            # Only classify as DoS if SYN flood is very clear (high SYN ratio)
            if syn_count >= 5 and total_packets >= 10:
                prediction = 'DoS'
        
        # Port Scanning (only if many unique ports)
        elif behavior['port_scan']:
            unique_ports = flow.get('unique_dst_ports', 0)
            # Only classify as Port Scanning if many unique ports
            if unique_ports >= 5:
                prediction = 'Port Scanning'
        
        # High packet rate -> DoS (only if very high rate AND short duration)
        elif behavior['high_packet_rate']:
            if flow['first_seen'] and flow['last_seen']:
                duration = flow['last_seen'] - flow['first_seen']
                total_packets = len(flow['packets'])
                # Only classify as DoS if very high rate in short time
                if duration > 0 and duration < 2.0 and total_packets > 50:
                    prediction = 'DoS'
        
        # High byte rate -> DDoS (only if extremely high)
        elif behavior['high_byte_rate']:
            if flow['first_seen'] and flow['last_seen']:
                duration = flow['last_seen'] - flow['first_seen']
                total_bytes = sum(p['size'] for p in flow['packets'])
                if duration > 0:
                    byte_rate = total_bytes / duration
                    # Only classify as DDoS if extremely high byte rate
                    if byte_rate > 5_000_000:  # >5MB/s
                        prediction = 'DDoS'
        
        # Do NOT use suspicious_pattern for rule-based override
        # Let ML model handle ambiguous cases
        
        rule_predictions.append(prediction)
    
    return rule_predictions

def hybrid_predict(model, features_df, flows, attack_behaviors):
    """
    Hybrid prediction combining rule-based detection and ML model.
    
    Decision priority:
    1. Rule-based attack detection (if clear pattern)
    2. ML model prediction
    
    This ensures attacks are detected even when feature distribution mismatch
    causes ML to predict Normal Traffic.
    
    Args:
        model: Trained XGBoost model
        features_df: DataFrame with features (potentially boosted)
        flows: List of flow dictionaries
        attack_behaviors: List of behavior dictionaries
    
    Returns:
        List of final predictions
    """
    # Step 1: Apply targeted feature boosting only for flows with clear attack behaviors
    # This helps ML model work better without being too aggressive
    boosted_features = boost_features_for_attacks(features_df, attack_behaviors)
    
    # Step 2: Get ML predictions on boosted features
    # The model will work better with properly adjusted features
    ml_predictions = predict_batch(model, boosted_features)
    
    # Step 3: Get rule-based predictions (conservative - only very clear patterns)
    rule_predictions = rule_based_detection(flows, attack_behaviors)
    
    # Step 4: Hybrid decision - balanced approach
    final_predictions = []
    for i, (rule_pred, ml_pred) in enumerate(zip(rule_predictions, ml_predictions)):
        # Only override ML if:
        # 1. Rule-based detected a VERY clear pattern AND
        # 2. ML predicted Normal (suggesting feature mismatch issue)
        if rule_pred is not None and ml_pred == 'Normal Traffic':
            # Rule-based override: very clear attack pattern that ML missed
            final_predictions.append(rule_pred)
        else:
            # Trust ML model prediction (it's well-trained and should work correctly)
            final_predictions.append(ml_pred)
    
    return final_predictions

# ============================================
# PCAP Extraction Functions (for Hybrid Detection)
# ============================================

def extract_flows_from_pcap(pcap_path):
    """
    Extract network flows from PCAP file using Python (scapy).
    
    This function parses PCAP packets and groups them into flows based on
    5-tuple: (src_ip, dst_ip, src_port, dst_port, protocol).
    Also captures TCP flags for rule-based detection.
    
    Args:
        pcap_path: Path to the PCAP file
    
    Returns:
        List of flow dictionaries, each containing packet information
    """
    if not SCAPY_AVAILABLE:
        raise ImportError(
            "scapy is required for PCAP parsing. Install with: pip install scapy"
        )
    
    packets = rdpcap(str(pcap_path))
    
    # Flow key: (src_ip, dst_ip, src_port, dst_port, protocol)
    # Flow data: list of packets with timestamps and sizes
    flows = defaultdict(lambda: {
        'packets': [],
        'fwd_packets': [],  # Source -> Destination
        'bwd_packets': [],  # Destination -> Source
        'first_seen': None,
        'last_seen': None,
        'src_ip': None,
        'dst_ip': None,
        'src_port': None,
        'dst_port': None,
        'protocol': None,
        'syn_count': 0,  # TCP SYN flags for SYN flood detection
        'ack_count': 0,  # TCP ACK flags
        'fin_count': 0,  # TCP FIN flags
        'psh_count': 0,  # TCP PSH flags
        'unique_dst_ports': set(),  # For port scanning detection
        'packet_timestamps': []  # For high packet rate detection
    })
    
    for packet in packets:
        # Extract IP layer
        if IP not in packet:
            continue
        
        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        
        # Extract transport layer (TCP/UDP)
        protocol = 0  # Default for non-TCP/UDP
        src_port = 0
        dst_port = 0
        syn_flag = False
        ack_flag = False
        fin_flag = False
        psh_flag = False
        
        if TCP in packet:
            protocol = 6  # TCP
            tcp_layer = packet[TCP]
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
            # Extract TCP flags
            flags = tcp_layer.flags
            syn_flag = bool(flags & 0x02)  # SYN flag
            ack_flag = bool(flags & 0x10)  # ACK flag
            fin_flag = bool(flags & 0x01)  # FIN flag
            psh_flag = bool(flags & 0x08)  # PSH flag
        elif UDP in packet:
            protocol = 17  # UDP
            udp_layer = packet[UDP]
            src_port = udp_layer.sport
            dst_port = udp_layer.dport
        elif ICMP in packet:
            protocol = 1  # ICMP
            # ICMP doesn't have ports, use type/code as identifier
            src_port = 0
            dst_port = 0
        else:
            continue  # Skip non-TCP/UDP/ICMP packets
        
        # Create flow key (bidirectional: smaller IP first)
        if src_ip < dst_ip:
            flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
            direction = 'fwd'
        else:
            flow_key = (dst_ip, src_ip, dst_port, src_port, protocol)
            direction = 'bwd'
        
        # Get packet timestamp and size
        timestamp = float(packet.time)
        packet_size = len(packet)
        
        # Update flow
        flow = flows[flow_key]
        if flow['first_seen'] is None:
            flow['first_seen'] = timestamp
            flow['src_ip'] = src_ip if direction == 'fwd' else dst_ip
            flow['dst_ip'] = dst_ip if direction == 'fwd' else src_ip
            flow['src_port'] = src_port if direction == 'fwd' else dst_port
            flow['dst_port'] = dst_port if direction == 'fwd' else src_port
            flow['protocol'] = protocol
        
        flow['last_seen'] = timestamp
        flow['packet_timestamps'].append(timestamp)
        
        # Track TCP flags
        if protocol == 6:  # TCP
            if syn_flag:
                flow['syn_count'] += 1
            if ack_flag:
                flow['ack_count'] += 1
            if fin_flag:
                flow['fin_count'] += 1
            if psh_flag:
                flow['psh_count'] += 1
        
        # Track unique destination ports for port scanning detection
        if direction == 'fwd':
            flow['unique_dst_ports'].add(dst_port)
        
        flow['packets'].append({
            'timestamp': timestamp,
            'size': packet_size,
            'direction': direction
        })
        
        if direction == 'fwd':
            flow['fwd_packets'].append({
                'timestamp': timestamp,
                'size': packet_size
            })
        else:
            flow['bwd_packets'].append({
                'timestamp': timestamp,
                'size': packet_size
            })
    
    # Convert sets to counts for easier processing
    for flow in flows.values():
        flow['unique_dst_ports'] = len(flow['unique_dst_ports'])
    
    return list(flows.values())

def calculate_flow_features(flow):
    """
    Calculate CICIDS2017-style features from a flow.
    
    Maps extracted flow data to the 52 features used by the model.
    Some features may be approximated if exact calculation isn't possible.
    
    Args:
        flow: Flow dictionary with packet information
    
    Returns:
        Dictionary with 52 features matching FEATURES list
    """
    # Extract packet lists
    fwd_packets = flow['fwd_packets']
    bwd_packets = flow['bwd_packets']
    all_packets = flow['packets']
    
    # Sort packets by timestamp
    all_packets.sort(key=lambda x: x['timestamp'])
    fwd_packets.sort(key=lambda x: x['timestamp'])
    bwd_packets.sort(key=lambda x: x['timestamp'])
    
    # Flow duration (in microseconds)
    if flow['first_seen'] and flow['last_seen']:
        flow_duration = (flow['last_seen'] - flow['first_seen']) * 1_000_000  # Convert to microseconds
    else:
        flow_duration = 0
    
    # Forward packet statistics
    fwd_sizes = [p['size'] for p in fwd_packets] if fwd_packets else [0]
    total_fwd_packets = len(fwd_packets)
    total_fwd_length = sum(fwd_sizes)
    fwd_packet_length_max = max(fwd_sizes) if fwd_sizes else 0
    fwd_packet_length_min = min(fwd_sizes) if fwd_sizes else 0
    fwd_packet_length_mean = np.mean(fwd_sizes) if fwd_sizes else 0
    fwd_packet_length_std = np.std(fwd_sizes) if len(fwd_sizes) > 1 else 0
    
    # Backward packet statistics
    bwd_sizes = [p['size'] for p in bwd_packets] if bwd_packets else [0]
    total_bwd_packets = len(bwd_packets)
    total_bwd_length = sum(bwd_sizes)
    bwd_packet_length_max = max(bwd_sizes) if bwd_sizes else 0
    bwd_packet_length_min = min(bwd_sizes) if bwd_sizes else 0
    bwd_packet_length_mean = np.mean(bwd_sizes) if bwd_sizes else 0
    bwd_packet_length_std = np.std(bwd_sizes) if len(bwd_sizes) > 1 else 0
    
    # All packet statistics
    all_sizes = [p['size'] for p in all_packets] if all_packets else [0]
    min_packet_length = min(all_sizes) if all_sizes else 0
    max_packet_length = max(all_sizes) if all_sizes else 0
    packet_length_mean = np.mean(all_sizes) if all_sizes else 0
    packet_length_std = np.std(all_sizes) if len(all_sizes) > 1 else 0
    packet_length_variance = np.var(all_sizes) if len(all_sizes) > 1 else 0
    average_packet_size = packet_length_mean
    
    # Flow rates (bytes/s and packets/s)
    if flow_duration > 0:
        flow_bytes_per_sec = (total_fwd_length + total_bwd_length) / (flow_duration / 1_000_000)
        flow_packets_per_sec = len(all_packets) / (flow_duration / 1_000_000)
    else:
        flow_bytes_per_sec = 0
        flow_packets_per_sec = 0
    
    # Inter-arrival times (IAT) - time between consecutive packets
    all_timestamps = [p['timestamp'] for p in all_packets]
    fwd_timestamps = [p['timestamp'] for p in fwd_packets]
    bwd_timestamps = [p['timestamp'] for p in bwd_packets]
    
    # Calculate IATs
    flow_iat = np.diff(all_timestamps) if len(all_timestamps) > 1 else [0]
    fwd_iat = np.diff(fwd_timestamps) if len(fwd_timestamps) > 1 else [0]
    bwd_iat = np.diff(bwd_timestamps) if len(bwd_timestamps) > 1 else [0]
    
    # IAT statistics (convert to microseconds)
    flow_iat_mean = np.mean(flow_iat) * 1_000_000 if len(flow_iat) > 0 else 0
    flow_iat_std = np.std(flow_iat) * 1_000_000 if len(flow_iat) > 1 else 0
    flow_iat_max = np.max(flow_iat) * 1_000_000 if len(flow_iat) > 0 else 0
    flow_iat_min = np.min(flow_iat) * 1_000_000 if len(flow_iat) > 0 else 0
    
    fwd_iat_total = np.sum(fwd_iat) * 1_000_000 if len(fwd_iat) > 0 else 0
    fwd_iat_mean = np.mean(fwd_iat) * 1_000_000 if len(fwd_iat) > 0 else 0
    fwd_iat_std = np.std(fwd_iat) * 1_000_000 if len(fwd_iat) > 1 else 0
    fwd_iat_max = np.max(fwd_iat) * 1_000_000 if len(fwd_iat) > 0 else 0
    fwd_iat_min = np.min(fwd_iat) * 1_000_000 if len(fwd_iat) > 0 else 0
    
    bwd_iat_total = np.sum(bwd_iat) * 1_000_000 if len(bwd_iat) > 0 else 0
    bwd_iat_mean = np.mean(bwd_iat) * 1_000_000 if len(bwd_iat) > 0 else 0
    bwd_iat_std = np.std(bwd_iat) * 1_000_000 if len(bwd_iat) > 1 else 0
    bwd_iat_max = np.max(bwd_iat) * 1_000_000 if len(bwd_iat) > 0 else 0
    bwd_iat_min = np.min(bwd_iat) * 1_000_000 if len(bwd_iat) > 0 else 0
    
    # Header lengths (approximate: IP header = 20 bytes, TCP header = 20-60 bytes)
    # Assumption: Average TCP header is ~32 bytes, UDP header is 8 bytes
    if flow['protocol'] == 6:  # TCP
        avg_header_size = 20 + 32  # IP + TCP
    elif flow['protocol'] == 17:  # UDP
        avg_header_size = 20 + 8  # IP + UDP
    else:
        avg_header_size = 20  # IP only
    
    fwd_header_length = total_fwd_packets * avg_header_size
    bwd_header_length = total_bwd_packets * avg_header_size
    
    # Packets per second (forward and backward)
    if flow_duration > 0:
        fwd_packets_per_sec = total_fwd_packets / (flow_duration / 1_000_000)
        bwd_packets_per_sec = total_bwd_packets / (flow_duration / 1_000_000)
    else:
        fwd_packets_per_sec = 0
        bwd_packets_per_sec = 0
    
    # TCP flags (extracted from actual TCP headers in extract_flows_from_pcap)
    fin_flag_count = flow.get('fin_count', 0)
    psh_flag_count = flow.get('psh_count', 0)
    ack_flag_count = flow.get('ack_count', 0)
    
    # If flags weren't captured, use approximations
    if flow['protocol'] == 6 and fwd_packets and ack_flag_count == 0:
        ack_flag_count = max(total_fwd_packets, total_bwd_packets)
        psh_flag_count = int(total_fwd_packets * 0.3)
        if len(fwd_packets) > 0 and len(bwd_packets) > 0:
            fin_flag_count = 1
    
    # TCP window sizes (approximation - not directly available from packets)
    # Assumption: Default window sizes
    init_win_bytes_forward = 65535  # Default TCP window
    init_win_bytes_backward = 65535
    
    # Subflow and segment features (approximations)
    subflow_fwd_bytes = total_fwd_length  # Approximate: all forward bytes
    act_data_pkt_fwd = total_fwd_packets  # Approximate: all packets have data
    min_seg_size_forward = fwd_packet_length_min if fwd_packets else 0
    
    # Active/Idle times (approximations based on packet timestamps)
    # Active: time between first and last packet
    # Idle: gaps between packets
    if len(all_timestamps) > 1:
        active_times = []
        idle_times = []
        
        for i in range(len(all_timestamps) - 1):
            gap = (all_timestamps[i+1] - all_timestamps[i]) * 1_000_000  # microseconds
            if gap < 1_000_000:  # Less than 1 second = active
                active_times.append(gap)
            else:  # More than 1 second = idle
                idle_times.append(gap)
        
        active_mean = np.mean(active_times) if active_times else 0
        active_max = np.max(active_times) if active_times else 0
        active_min = np.min(active_times) if active_times else 0
        
        idle_mean = np.mean(idle_times) if idle_times else 0
        idle_max = np.max(idle_times) if idle_times else 0
        idle_min = np.min(idle_times) if idle_times else 0
    else:
        active_mean = active_max = active_min = 0
        idle_mean = idle_max = idle_min = 0
    
    # Build feature dictionary matching FEATURES order
    features = {
        'Destination Port': flow['dst_port'],
        'Flow Duration': flow_duration,
        'Total Fwd Packets': total_fwd_packets,
        'Total Length of Fwd Packets': total_fwd_length,
        'Fwd Packet Length Max': fwd_packet_length_max,
        'Fwd Packet Length Min': fwd_packet_length_min,
        'Fwd Packet Length Mean': fwd_packet_length_mean,
        'Fwd Packet Length Std': fwd_packet_length_std,
        'Bwd Packet Length Max': bwd_packet_length_max,
        'Bwd Packet Length Min': bwd_packet_length_min,
        'Bwd Packet Length Mean': bwd_packet_length_mean,
        'Bwd Packet Length Std': bwd_packet_length_std,
        'Flow Bytes/s': flow_bytes_per_sec,
        'Flow Packets/s': flow_packets_per_sec,
        'Flow IAT Mean': flow_iat_mean,
        'Flow IAT Std': flow_iat_std,
        'Flow IAT Max': flow_iat_max,
        'Flow IAT Min': flow_iat_min,
        'Fwd IAT Total': fwd_iat_total,
        'Fwd IAT Mean': fwd_iat_mean,
        'Fwd IAT Std': fwd_iat_std,
        'Fwd IAT Max': fwd_iat_max,
        'Fwd IAT Min': fwd_iat_min,
        'Bwd IAT Total': bwd_iat_total,
        'Bwd IAT Mean': bwd_iat_mean,
        'Bwd IAT Std': bwd_iat_std,
        'Bwd IAT Max': bwd_iat_max,
        'Bwd IAT Min': bwd_iat_min,
        'Fwd Header Length': fwd_header_length,
        'Bwd Header Length': bwd_header_length,
        'Fwd Packets/s': fwd_packets_per_sec,
        'Bwd Packets/s': bwd_packets_per_sec,
        'Min Packet Length': min_packet_length,
        'Max Packet Length': max_packet_length,
        'Packet Length Mean': packet_length_mean,
        'Packet Length Std': packet_length_std,
        'Packet Length Variance': packet_length_variance,
        'FIN Flag Count': fin_flag_count,
        'PSH Flag Count': psh_flag_count,
        'ACK Flag Count': ack_flag_count,
        'Average Packet Size': average_packet_size,
        'Subflow Fwd Bytes': subflow_fwd_bytes,
        'Init_Win_bytes_forward': init_win_bytes_forward,
        'Init_Win_bytes_backward': init_win_bytes_backward,
        'act_data_pkt_fwd': act_data_pkt_fwd,
        'min_seg_size_forward': min_seg_size_forward,
        'Active Mean': active_mean,
        'Active Max': active_max,
        'Active Min': active_min,
        'Idle Mean': idle_mean,
        'Idle Max': idle_max,
        'Idle Min': idle_min
    }
    
    return features

def extract_features_from_pcap(pcap_path):
    """
    Extract flows from PCAP and calculate features for each flow.
    
    Args:
        pcap_path: Path to PCAP file
    
    Returns:
        Tuple of (pandas.DataFrame with 52 features, list of flow dictionaries)
    """
    # Extract flows
    flows = extract_flows_from_pcap(pcap_path)
    
    if not flows:
        raise ValueError("No valid flows found in PCAP file. Ensure the file contains TCP/UDP/ICMP packets.")
    
    # Calculate features for each flow
    features_list = []
    valid_flows = []
    for flow in flows:
        try:
            features = calculate_flow_features(flow)
            features_list.append(features)
            valid_flows.append(flow)
        except Exception as e:
            # Skip flows that cause errors
            continue
    
    if not features_list:
        raise ValueError("Could not extract features from any flows in the PCAP file.")
    
    # Create DataFrame with exact FEATURES order
    df = pd.DataFrame(features_list)
    
    # Ensure all features are present and in correct order
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0  # Fill missing features with 0
    
    # Reorder columns to match FEATURES exactly
    df = df[FEATURES]
    
    return df, valid_flows

def preprocess_flows(df):
    """
    Preprocess extracted flows for prediction.
    
    - Select only the predefined FEATURES
    - Handle NaN and Inf values safely
    - Do NOT scale (model was trained on unscaled data)
    """
    # Select only required features
    missing_features = set(FEATURES) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    features_df = df[FEATURES].copy()
    
    # Replace inf with NaN, then fill NaN with 0
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)
    
    return features_df

def tab_manual_input(model, explainer):
    """Live detection tab - test the model on network traffic."""
    st.header("🔍 Live Detection")

    st.info("""
    **What is this?**
    Upload a PCAP file to analyze network traffic, or test the model with manual feature input.
    The system uses hybrid detection (rule-based + ML) to ensure attacks are properly detected.
    """)

    # PCAP Upload Section
    st.markdown("---")
    st.subheader("📁 PCAP File Upload")
    
    uploaded_pcap = st.file_uploader(
        "Upload PCAP File",
        type=['pcap', 'pcapng'],
        help="Upload a network packet capture file (.pcap or .pcapng format) for hybrid detection",
        key="pcap_uploader"
    )
    
    if uploaded_pcap is not None:
        # Ensure we stay on Live Detection tab after upload
        st.session_state.selected_tab = "🔍 Live Detection"
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Save uploaded PCAP to temporary file
                pcap_path = Path(temp_dir) / uploaded_pcap.name
                with open(pcap_path, "wb") as f:
                    f.write(uploaded_pcap.getbuffer())
                
                st.success(f"✅ Uploaded **{uploaded_pcap.name}** ({uploaded_pcap.size / 1024 / 1024:.2f} MB)")
                
                # Step 1: Extract flows and calculate features from PCAP
                with st.spinner("🔄 Extracting network flows and calculating features (this may take a minute)..."):
                    features_df, flows = extract_features_from_pcap(pcap_path)
                
                st.success(f"✅ Extracted **{len(features_df):,}** network flows")
                
                # Step 2: Detect attack behaviors for feature boosting
                with st.spinner("🔍 Analyzing attack patterns..."):
                    attack_behaviors = detect_attack_behaviors(flows)
                
                # Step 3: Preprocess flows (handle NaN/Inf)
                with st.spinner("🔧 Preprocessing flows..."):
                    features_df = preprocess_flows(features_df)
                
                # Step 4: Hybrid prediction (rule-based + ML with adaptive boosting)
                # This compensates for feature extraction mismatch with CICFlowMeter
                # Feature boosting is applied inside hybrid_predict if needed
                with st.spinner("🤖 Running hybrid detection (rule-based + AI with adaptive boosting)..."):
                    predictions = hybrid_predict(model, features_df, flows, attack_behaviors)
                
                # Step 6: Display results
                st.markdown("---")
                st.subheader("📊 Detection Results")
                
                # Calculate statistics
                total_flows = len(predictions)
                attacks = sum(1 for p in predictions if p != 'Normal Traffic')
                normal_count = total_flows - attacks
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Flows", f"{total_flows:,}")
                with col2:
                    st.metric("Attacks Detected", f"{attacks:,}", delta=f"{attacks/total_flows*100:.1f}%")
                with col3:
                    st.metric("Normal Traffic", f"{normal_count:,}")
                
                # Attack distribution pie chart
                st.markdown("---")
                st.subheader("Attack Distribution")
                
                pred_counts = pd.Series(predictions).value_counts()
                
                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    color=pred_counts.index,
                    color_discrete_map=ATTACK_COLORS,
                    hole=0.4,
                    title="Distribution of Detected Traffic Types"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Step 7: Explainable AI - Representative Samples
                if attacks > 0:
                    st.markdown("---")
                    st.subheader("🧠 Why was this attack detected?")
                    st.caption("For each detected attack type, showing one representative flow with top influential features:")
                    
                    # Get unique attack types (excluding Normal Traffic)
                    unique_attacks = [p for p in set(predictions) if p != "Normal Traffic"]
                    
                    for attack in unique_attacks:
                        try:
                            # Find first occurrence of this attack type
                            attack_indices = [i for i, p in enumerate(predictions) if p == attack]
                            if not attack_indices:
                                continue
                            
                            # Select first representative sample
                            sample_idx = attack_indices[0]
                            sample_df = features_df.iloc[[sample_idx]]
                            
                            # Get SHAP values for this sample
                            shap_vals = explainer.shap_values(sample_df)
                            class_idx = LABEL_MAPPING[attack]
                            
                            # Handle both list format and 3D numpy array format
                            if isinstance(shap_vals, list):
                                values = shap_vals[class_idx][0]
                            else:
                                # 3D numpy array format: (samples, features, classes)
                                values = shap_vals[0, :, class_idx]
                            
                            st.markdown(f"### 🔴 {attack}")
                            st.caption(f"Flow #{sample_idx + 1} from PCAP file")
                            
                            # Get top 5 features by absolute SHAP value
                            top_feats = pd.Series(values, index=FEATURES)\
                                .sort_values(key=abs, ascending=False).head(5)
                            
                            st.markdown("**Top 5 Influential Features:**")
                            # Display features with direction of influence
                            for feat, shap_val in top_feats.items():
                                direction = "↑" if shap_val > 0 else "↓"
                                influence = "increases risk" if shap_val > 0 else "decreases risk"
                                st.markdown(f"- **{feat}** ({direction} {influence})")
                            
                            # Recommended Actions section
                            st.markdown("**🛠️ Recommended Response Actions:**")
                            if attack in ATTACK_ACTIONS:
                                for action in ATTACK_ACTIONS[attack]:
                                    st.markdown(f"- {action}")
                            else:
                                st.info("No specific actions defined for this attack type.")
                            
                            st.markdown("---")
                            
                        except Exception as e:
                            st.warning(f"Could not generate explanation for {attack}: {str(e)}")
                else:
                    st.info("✅ No attacks detected. All flows are classified as Normal Traffic.")
                
            except ImportError as e:
                st.error(f"❌ **Missing dependency:** {str(e)}")
                st.info("Install scapy with: `pip install scapy`")
            except ValueError as e:
                st.error(f"❌ **Invalid PCAP file:** {str(e)}")
            except Exception as e:
                st.error(f"❌ **Error processing PCAP file:** {str(e)}")
                st.exception(e)
    
    st.markdown("---")
    st.subheader("📝 Manual Feature Input (Alternative)")
    
    # Initialize session state for all feature inputs if not exists
    for feat in FEATURES:
        if f"input_{feat}" not in st.session_state:
            st.session_state[f"input_{feat}"] = 0.0

    # Load Example button
    demo_df = load_demo_samples()
    if demo_df is not None:
        if st.button("🎲 Load Random Example", use_container_width=True):
            # Pick a random sample
            sample = demo_df.sample(n=1).iloc[0]
            # Update ALL widget keys directly in session state
            for feat in FEATURES:
                st.session_state[f"input_{feat}"] = float(sample[feat])
            st.session_state['manual_example_type'] = sample.get('Attack Type', 'Unknown')
            st.rerun()

        # Show what type of example was loaded
        if 'manual_example_type' in st.session_state:
            example_type = st.session_state['manual_example_type']
            if example_type == 'Normal Traffic':
                st.success(f"📥 Loaded example: **{example_type}**")
            else:
                st.warning(f"📥 Loaded example: **{example_type}** (attack)")

    st.markdown("---")

    # Input form
    st.subheader("Feature Values")
    st.caption("Enter values for the network flow features below")

    # Create input fields in columns
    input_values = {}

    # Key features to show prominently
    key_features = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packet Length Mean',
        'FIN Flag Count', 'PSH Flag Count', 'ACK Flag Count'
    ]

    st.markdown("**Key Features (most important):**")
    cols = st.columns(3)
    for i, feat in enumerate(key_features):
        with cols[i % 3]:
            input_values[feat] = st.number_input(
                feat,
                help=FEATURE_EXPLANATIONS.get(feat, ''),
                key=f"input_{feat}",
                format="%.2f"
            )

    # Other features in expander
    with st.expander("📋 All Other Features (43 more)", expanded=False):
        other_features = [f for f in FEATURES if f not in key_features]
        cols = st.columns(3)
        for i, feat in enumerate(other_features):
            with cols[i % 3]:
                input_values[feat] = st.number_input(
                    feat,
                    help=FEATURE_EXPLANATIONS.get(feat, ''),
                    key=f"input_{feat}",
                    format="%.2f"
                )

    st.markdown("---")

    # Predict button
    if st.button("🔍 Analyze This Traffic", type="primary", use_container_width=True):
        # Create dataframe from inputs - get values directly from session state
        input_data = {feat: st.session_state[f"input_{feat}"] for feat in FEATURES}
        input_df = pd.DataFrame([input_data])[FEATURES]

        # Make prediction
        prediction = predict_single(model, input_df)

        st.markdown("### Prediction Result")
        render_prediction_result(prediction)

        # Explainable AI section - only for attacks
        if prediction != "Normal Traffic":
            try:
                # Get SHAP values
                shap_vals = explainer.shap_values(input_df)
                class_idx = LABEL_MAPPING[prediction]

                # Handle both list format and 3D numpy array format
                if isinstance(shap_vals, list):
                    values = shap_vals[class_idx][0]
                else:
                    # 3D numpy array format: (samples, features, classes)
                    values = shap_vals[0, :, class_idx]

                st.markdown("---")
                st.subheader("🧠 Why did the model predict this?")
                st.caption("Top 5 most influential features that led to this prediction:")

                # Get top 5 features by absolute SHAP value
                top_feats = pd.Series(values, index=FEATURES)\
                    .sort_values(key=abs, ascending=False).head(5)

                # Display features with direction of influence
                for feat, shap_val in top_feats.items():
                    direction = "+" if shap_val > 0 else "-"
                    st.markdown(f"- **{feat}** ({direction})")

                # Recommended Actions section
                st.markdown("---")
                st.subheader("🛠 Recommended Response Actions")
                if prediction in ATTACK_ACTIONS:
                    for action in ATTACK_ACTIONS[prediction]:
                        st.markdown(f"- {action}")
            except Exception as e:
                st.warning(f"Could not generate explanation: {str(e)}")

# ============================================
# Tab 3: Batch Analysis (CSV Upload)
# ============================================

def tab_batch_analysis(model, explainer):
    """Batch analysis tab content."""
    st.header("📁 Batch Analysis (CSV Upload)")

    st.info("""
    **What is this?**
    Upload a CSV file with network traffic data to analyze many samples at once.
    The CSV must have the same 52 features as the training data.
    """)

    # Download template button
    template_df = pd.DataFrame(columns=FEATURES)
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        "📥 Download Empty Template CSV",
        template_csv,
        "template.csv",
        "text/csv"
    )

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Must contain the 52 network flow features. 'Attack Type' column is optional for accuracy check."
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded **{len(df):,}** samples")

            has_labels = 'Attack Type' in df.columns

            if has_labels:
                features = df.drop('Attack Type', axis=1)
                actual_labels = df['Attack Type'].tolist()
            else:
                features = df
                actual_labels = None

            # Validate columns
            missing_cols = set(FEATURES) - set(features.columns)
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                return

            features = features[FEATURES]

            # Make predictions
            with st.spinner("🔄 Analyzing traffic..."):
                predictions = predict_batch(model, features)

            # Results
            results_df = df.copy()
            results_df['Predicted'] = predictions

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Attack Distribution")
                pred_counts = pd.Series(predictions).value_counts()

                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    color=pred_counts.index,
                    color_discrete_map=ATTACK_COLORS,
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Summary")

                total = len(predictions)
                attacks = sum(1 for p in predictions if p != 'Normal Traffic')

                st.metric("Total Samples", f"{total:,}")
                st.metric("Attacks Detected", f"{attacks:,}", delta=f"{attacks/total*100:.1f}% of traffic")
                st.metric("Normal Traffic", f"{total - attacks:,}")

                if has_labels:
                    correct = sum(1 for p, a in zip(predictions, actual_labels) if p == a)
                    st.metric("Accuracy", f"{correct/total*100:.2f}%")

            st.subheader("Detailed Results")
            if has_labels:
                results_df['Correct?'] = ['✓' if p == a else '✗' for p, a in zip(predictions, actual_labels)]

            st.dataframe(results_df, use_container_width=True, height=300)

            csv = results_df.to_csv(index=False)
            st.download_button("📥 Download Results", csv, "results.csv", "text/csv", use_container_width=True)

            # Explainable AI - Representative Samples
            st.markdown("---")
            st.subheader("🧠 Explainable AI – Representative Samples")
            st.caption("For each detected attack type, showing one representative sample with top 3 influential features:")

            # Get unique attack types (excluding Normal Traffic)
            unique_attacks = [p for p in set(predictions) if p != "Normal Traffic"]

            if unique_attacks:
                for attack in unique_attacks:
                    try:
                        # Find first occurrence of this attack type
                        attack_indices = [i for i, p in enumerate(predictions) if p == attack]
                        if not attack_indices:
                            continue

                        # Select first representative sample
                        sample_idx = attack_indices[0]
                        sample_df = features.iloc[[sample_idx]]

                        # Get SHAP values for this sample
                        shap_vals = explainer.shap_values(sample_df)
                        class_idx = LABEL_MAPPING[attack]

                        # Handle both list format and 3D numpy array format
                        if isinstance(shap_vals, list):
                            values = shap_vals[class_idx][0]
                        else:
                            # 3D numpy array format: (samples, features, classes)
                            values = shap_vals[0, :, class_idx]

                        st.markdown(f"### 🔴 {attack}")
                        st.caption(f"Sample #{sample_idx + 1} from uploaded data")

                        # Get top 3 features by absolute SHAP value
                        top_feats = pd.Series(values, index=FEATURES)\
                            .sort_values(key=abs, ascending=False).head(3)

                        # Display features with direction of influence
                        for feat, shap_val in top_feats.items():
                            direction = "+" if shap_val > 0 else "-"
                            st.markdown(f"- **{feat}** ({direction})")

                    except Exception as e:
                        st.warning(f"Could not generate explanation for {attack}: {str(e)}")
            else:
                st.info("No attacks detected in the uploaded data. All samples are classified as Normal Traffic.")

        except Exception as e:
            st.error(f"Error: {str(e)}")


# ============================================
# Tab 5: How It Works
# ============================================

def tab_how_it_works():
    """Educational content about the system."""
    st.header("📚 How It Works")

    # The Pipeline
    st.subheader("1. The Detection Pipeline")

    st.markdown("""
    ```
    Real World:
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   Network    │ ──► │  Capture     │ ──► │  Extract     │ ──► │   ML Model   │
    │   Traffic    │     │  Packets     │     │  Features    │     │  Predicts    │
    │              │     │  (Wireshark) │     │ (CICFlowMeter)│    │              │
    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                              │                     │                    │
                              ▼                     ▼                    ▼
                          .pcap file          52 numbers            Attack or
                        (raw packets)        per connection         Normal?
    ```
    """)

    st.info("""
    **In simple terms:**
    1. Network traffic is captured (like recording all phone calls)
    2. Each "conversation" (connection) is analyzed to extract 52 statistics
    3. The AI model looks at these numbers and decides: attack or normal?
    """)

    st.markdown("---")

    # Attack Types
    st.subheader("2. Attack Types We Detect")

    for attack, desc in ATTACK_DESCRIPTIONS.items():
        color = ATTACK_COLORS[attack]
        icon = "🟢" if attack == "Normal Traffic" else "🔴"

        with st.expander(f"{icon} {attack}"):
            st.markdown(desc)

            # Add real-world example
            examples = {
                'Normal Traffic': 'Example: You browsing Instagram, watching YouTube, sending email.',
                'DoS': 'Example: Attacker sends 1 million requests per second to crash your website.',
                'DDoS': 'Example: The 2016 Dyn attack took down Twitter, Netflix, Reddit using 100,000 hacked cameras.',
                'Port Scanning': 'Example: Hacker runs "nmap" to see if port 22 (SSH) is open on your server.',
                'Brute Force': 'Example: Trying passwords like "password123", "admin", "12345" until one works.',
                'Web Attacks': 'Example: Typing `\' OR 1=1--` in a login form to bypass authentication.',
                'Bots': 'Example: Your computer secretly mining Bitcoin for hackers without you knowing.'
            }
            st.caption(examples.get(attack, ''))

    st.markdown("---")

    # Features
    st.subheader("3. What Features Does the Model Use?")

    st.markdown("""
    The model looks at **52 statistics** about each network connection.
    Think of it like a doctor checking vital signs - the AI checks "vital signs" of network traffic.
    """)

    # Group features by category
    feature_groups = {
        "📦 Packet Info": ['Total Fwd Packets', 'Fwd Packet Length Mean', 'Bwd Packet Length Mean'],
        "⏱️ Timing": ['Flow Duration', 'Flow IAT Mean', 'Active Mean'],
        "🚀 Speed": ['Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s'],
        "🚩 TCP Flags": ['FIN Flag Count', 'PSH Flag Count', 'ACK Flag Count'],
        "🚪 Connection": ['Destination Port', 'Init_Win_bytes_forward']
    }

    cols = st.columns(len(feature_groups))
    for i, (group_name, features) in enumerate(feature_groups.items()):
        with cols[i]:
            st.markdown(f"**{group_name}**")
            for feat in features:
                st.markdown(f"• {feat}")

    with st.expander("📋 View All 52 Features"):
        col1, col2 = st.columns(2)
        half = len(FEATURES) // 2

        with col1:
            for feat in FEATURES[:half]:
                st.markdown(f"**{feat}**")
                st.caption(FEATURE_EXPLANATIONS.get(feat, ''))
        with col2:
            for feat in FEATURES[half:]:
                st.markdown(f"**{feat}**")
                st.caption(FEATURE_EXPLANATIONS.get(feat, ''))

    st.markdown("---")

    # Dataset
    st.subheader("4. The Dataset: CICIDS2017")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Created by:** Canadian Institute for Cybersecurity
        **University:** University of New Brunswick, Canada
        **Year:** 2017

        **How they made it:**
        1. Built a fake network in their lab
        2. Ran normal traffic (browsing, email, etc.)
        3. Launched real attacks using hacking tools
        4. Recorded everything for 5 days
        5. Labeled each connection as attack or normal
        """)

    with col2:
        st.markdown("**Data Distribution:**")
        dist_data = {
            'Attack Type': list(LABEL_MAPPING.keys()),
            'Samples': [2095057, 193745, 128014, 90694, 9150, 2143, 1948]
        }
        dist_df = pd.DataFrame(dist_data)

        fig = px.bar(
            dist_df,
            x='Attack Type',
            y='Samples',
            color='Attack Type',
            color_discrete_map=ATTACK_COLORS,
            log_y=True
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# Sidebar
# ============================================

def render_sidebar():
    """Render sidebar with info."""
    with st.sidebar:
        st.header("ℹ️ Quick Info")

        st.markdown("""
        **Model:** XGBoost
        **Accuracy:** 99.90%
        **Dataset:** CICIDS2017
        **Features:** 52
        **Attack Types:** 6
        """)

        st.markdown("---")

        st.markdown("**Attack Types:**")
        for attack in LABEL_MAPPING.keys():
            icon = "🟢" if attack == "Normal Traffic" else "🔴"
            st.markdown(f"{icon} {attack}")

# ============================================
# Main App
# ============================================

def main():
    """Main application entry point."""
    render_header()
    render_sidebar()

    # Load resources
    try:
        model = load_model()
        explainer = load_explainer()
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        st.info("Make sure the model and explainer files are in the correct location.")
        st.code("""
Required files:
- models/xgboost.joblib
- models/shap_explainer.joblib
        """)
        return

    # Tab persistence fix: Use session state to maintain active tab
    # This prevents Streamlit from resetting to first tab after PCAP upload
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "🔍 Live Detection"  # Default to Live Detection
    
    # Custom tab implementation using radio buttons for full state control
    # This ensures tab selection persists across reruns (e.g., after file upload)
    tab_options = ["📚 How It Works", "🔍 Live Detection", "📁 Batch Analysis"]
    
    # Create horizontal radio buttons styled as tabs
    selected = st.radio(
        "Navigation",
        options=tab_options,
        index=tab_options.index(st.session_state.selected_tab) if st.session_state.selected_tab in tab_options else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="tab_selector"
    )
    
    # Update session state with selected tab
    st.session_state.selected_tab = selected
    
    # Render content based on selected tab
    if selected == "📚 How It Works":
        tab_how_it_works()
    elif selected == "🔍 Live Detection":
        tab_manual_input(model, explainer)
    elif selected == "📁 Batch Analysis":
        tab_batch_analysis(model, explainer)


if __name__ == "__main__":
    main()
