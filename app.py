import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import train_test_split

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
def load_scaler():
    """Load the RobustScaler."""
    scaler_path = Path(__file__).parent / "scalers" / "robust_scaler.joblib"
    return joblib.load(scaler_path)

@st.cache_data
def load_and_split_data():
    """
    Load data and split into train/test sets.
    Returns ONLY the test set (data the model hasn't seen during training).
    Uses the same random_state=42 as the notebook to get the exact same split.
    """
    data_path = Path(__file__).parent / "data" / "cicids2017_cleaned.csv"
    df = pd.read_csv(data_path)

    # Same split as in the notebook (70% train, 30% test, stratified, random_state=42)
    X = df.drop('Attack Type', axis=1)
    y = df['Attack Type']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Combine back into a dataframe
    test_df = X_test.copy()
    test_df['Attack Type'] = y_test

    return test_df

# ============================================
# Prediction Functions
# ============================================

def predict_single(model, scaler, features_df):
    """Make prediction for a single sample."""
    X_scaled = scaler.transform(features_df)
    prediction = model.predict(X_scaled)
    return REVERSE_MAPPING[prediction[0]]

def predict_batch(model, scaler, features_df):
    """Make predictions for multiple samples."""
    X_scaled = scaler.transform(features_df)
    predictions = model.predict(X_scaled)
    return [REVERSE_MAPPING[p] for p in predictions]

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
# Tab 1: Test on Unseen Data
# ============================================

def tab_test_detection(model, scaler, test_df):
    """Test detection on data the model has never seen."""
    st.header("🧪 Test on Unseen Data")

    # Explanation box
    st.info("""
    **What is this?**
    This tests the model on **real network traffic data it has NEVER seen before**.
    The data comes from the CICIDS2017 dataset (30% held out for testing).
    This proves the model can detect NEW attacks, not just memorize old ones.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Get a Sample")

        # Show available attack types
        st.markdown("**Available attack types in test data:**")
        attack_counts = test_df['Attack Type'].value_counts()
        for attack, count in attack_counts.items():
            color = "🟢" if attack == "Normal Traffic" else "🔴"
            st.markdown(f"{color} {attack}: {count:,} samples")

        st.markdown("---")

        # Random sample button
        if st.button("🎲 Get Random Test Sample", type="primary", use_container_width=True):
            sample = test_df.sample(n=1)
            st.session_state['test_sample'] = sample
            st.session_state['test_actual'] = sample['Attack Type'].values[0]

    with col2:
        st.subheader("Prediction Result")

        if 'test_sample' in st.session_state:
            sample = st.session_state['test_sample']
            actual = st.session_state['test_actual']

            # Get features only
            features = sample.drop('Attack Type', axis=1)

            # Make prediction
            prediction = predict_single(model, scaler, features)

            # Display result
            render_prediction_result(prediction, actual)

            # Show some key features
            st.markdown("---")
            with st.expander("📊 View Network Flow Details", expanded=False):
                st.markdown("**Key features of this network connection:**")

                key_features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets',
                               'Flow Bytes/s', 'FIN Flag Count', 'ACK Flag Count']

                for feat in key_features:
                    val = features[feat].values[0]
                    explanation = FEATURE_EXPLANATIONS.get(feat, '')
                    st.markdown(f"**{feat}**: `{val:,.2f}`")
                    st.caption(explanation)
        else:
            st.markdown("👆 Click **'Get Random Test Sample'** to test the model on unseen data")

# ============================================
# Tab 2: Manual Input
# ============================================

def tab_manual_input(model, scaler):
    """Manual input for custom testing."""
    st.header("✏️ Manual Input")

    st.info("""
    **What is this?**
    Enter your own network flow values to see what the model predicts.
    All 52 features must be filled in for the model to make a prediction.
    """)

    # Input form
    st.subheader("Feature Values")
    st.caption("Enter values for the network flow features below")

    # Get default values
    if 'manual_values' not in st.session_state:
        # Use median values as defaults
        st.session_state['manual_values'] = {feat: 0.0 for feat in FEATURES}

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
            default_val = st.session_state['manual_values'].get(feat, 0.0)
            input_values[feat] = st.number_input(
                feat,
                value=float(default_val),
                help=FEATURE_EXPLANATIONS.get(feat, ''),
                key=f"input_{feat}"
            )

    # Other features in expander
    with st.expander("📋 All Other Features (43 more)", expanded=False):
        other_features = [f for f in FEATURES if f not in key_features]
        cols = st.columns(3)
        for i, feat in enumerate(other_features):
            with cols[i % 3]:
                default_val = st.session_state['manual_values'].get(feat, 0.0)
                input_values[feat] = st.number_input(
                    feat,
                    value=float(default_val),
                    help=FEATURE_EXPLANATIONS.get(feat, ''),
                    key=f"input_{feat}"
                )

    st.markdown("---")

    # Predict button
    if st.button("🔍 Analyze This Traffic", type="primary", use_container_width=True):
        # Create dataframe from inputs
        input_df = pd.DataFrame([input_values])[FEATURES]

        # Make prediction
        prediction = predict_single(model, scaler, input_df)

        st.markdown("### Prediction Result")
        render_prediction_result(prediction)

# ============================================
# Tab 3: Batch Analysis (CSV Upload)
# ============================================

def tab_batch_analysis(model, scaler):
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
                predictions = predict_batch(model, scaler, features)

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

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ============================================
# Tab 4: Model Performance
# ============================================

def tab_model_performance():
    """Model performance tab content."""
    st.header("📈 Model Performance")

    st.info("""
    **What is this?**
    These are the results from training our XGBoost model on the CICIDS2017 dataset.
    The model was trained on 70% of the data and tested on 30% it had never seen.
    """)

    # Metrics explanation
    with st.expander("❓ What do these metrics mean?", expanded=False):
        st.markdown("""
        - **Accuracy**: Overall, how often is the model correct?
        - **Precision**: When it says "attack", how often is it right?
        - **Recall**: Of all real attacks, how many did it catch?
        - **F1 Score**: Balance between precision and recall
        """)

    # Overall metrics
    st.subheader("Overall Results")

    col1, col2, col3, col4 = st.columns(4)
    metrics = METRICS['XGBoost']

    col1.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
    col2.metric("Precision", f"{metrics['Precision']*100:.2f}%")
    col3.metric("Recall", f"{metrics['Recall']*100:.2f}%")
    col4.metric("F1 Score", f"{metrics['F1 Score']*100:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Confusion Matrix")
        st.caption("Shows what the model predicted vs. actual labels")

        fig = px.imshow(
            CONFUSION_MATRIX,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=CLASS_LABELS,
            y=CLASS_LABELS,
            color_continuous_scale="Blues",
            text_auto=True
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Per-Attack Metrics")

        metrics_df = pd.DataFrame(PER_CLASS_METRICS).T
        metrics_df = metrics_df.round(2)
        metrics_df.columns = ['Precision', 'Recall', 'F1', 'Samples']
        st.dataframe(metrics_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Training Resources")
        st.metric("Memory Used", f"{metrics['Memory (MB)']:.1f} MB")
        st.metric("Training Time", f"{metrics['Training Time (s)']:.1f} seconds")

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
        scaler = load_scaler()
        test_df = load_and_split_data()
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        st.info("Make sure the model, scaler, and data files are in the correct locations.")
        st.code("""
Required files:
- models/xgboost.joblib
- scalers/robust_scaler.joblib
- data/cicids2017_cleaned.csv
        """)
        return

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧪 Test Detection",
        "✏️ Manual Input",
        "📁 Batch Analysis",
        "📈 Model Performance",
        "📚 How It Works"
    ])

    with tab1:
        tab_test_detection(model, scaler, test_df)

    with tab2:
        tab_manual_input(model, scaler)

    with tab3:
        tab_batch_analysis(model, scaler)

    with tab4:
        tab_model_performance()

    with tab5:
        tab_how_it_works()

if __name__ == "__main__":
    main()
