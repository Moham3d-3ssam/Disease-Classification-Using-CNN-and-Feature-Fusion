import streamlit as st
import pandas as pd
import tensorflow as tf
import os
import base64
import pickle
import joblib
import numpy as np
import cv2

import pickle

with open("sc.pkl", "rb") as f:
    sc = pickle.load(f)

with open("orderLabels.pkl", "rb") as f:
    orderLabels = pickle.load(f)

with open("ensemble_clf.pki", "rb") as f:
    ensemble_clf = pickle.load(f)

with open("xgb_clf.pki", "rb") as f:
    xgb_clf = pickle.load(f)

with open("lgb_clf.pki", "rb") as f:
    lgb_clf = pickle.load(f)

cnn_tf_model = tf.keras.models.load_model("cnn_tf_model.keras")

# st.set_page_config(layout = "wide")
st.markdown("""
<style>
    .main .block-container {
        padding-top: 4rem;
    }
    div.stRadio > label {
        font-size: 1.5rem; /* Increased font size for choices */
    }
    div.stSlider div[data-baseweb="slider"] {
        height: 10px; /* Adjust this to make the scroll bar bigger */
    }
    .prediction-box {
        background-color: #262730;
        color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        text-align: left;
    }
    .final-card {
        display: flex;
        flex-direction: row;
        background-color: #262730;
        color: #ffffff;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin-top: 2rem;
        align-items: center;
    }
    .final-card-image {
        flex: 1;
        text-align: center;
        padding-right: 2rem;
    }
    .final-card-info {
        flex: 2;
        padding-left: 2rem;
        border-left: 1px solid #444;
    }
    .final-card-info h3 {
        margin-top: 0;
        color: #ddd;
    }
    .st-emotion-cache-16ya5n7 h3 {
        padding: 0;
    }
    .uploaded-image-circle {
        border-radius: 50%;
        width: 150px;
        height: 150px;
        object-fit: cover;
        border: 4px solid #fff;
    }
    .percentage-circle {
        position: relative;
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: radial-gradient(closest-side, #262730 79%, transparent 80% 100%),
                    conic-gradient(#23d17c var(--percentage), #444 0);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 1rem;
    }
    /* Adjusted CSS to center the text within the circle and keep it on one line */
    .percentage-circle h2 {
        color: #23d17c;
        font-size: 3rem;
        white-space: nowrap;
        margin: 0;
    }
    .percentage-circle .percentage-text {
        color: #23d17c;
        font-size: 3rem;
        white-space: nowrap;
        margin: 0;
    }
    /* Flex container for two info items in one row */
    .info-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px; /* Space between rows */
    }
    .info-item {
        flex: 1; /* Each item takes equal space */
        margin-right: 10px; /* Space between items in a row */
    }
    .info-item:last-child {
        margin-right: 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Loads the CSV, cleans the data, and prepares the questions."""
    if not os.path.exists("metadata.csv"):
        st.error("metadata.csv not found in the current directory.")
        return None, None

    df = pd.read_csv("metadata.csv")

    columns_to_exclude = ['background_father', 'background_mother', 'img_id', 'diagnostic']
    columns_to_analyze = [col for col in df.columns if col not in columns_to_exclude]

    desired_order = [
        'name',
        'age',
        'gender',
        'race',
        'region',
        'diameter_1',
        'diameter_2',
        'smoke',
        'drink',
        'skin_cancer_history',
        'cancer_history',
        'pesticide',
        'biopsed',
        'grew',
        'hurt',
        'bleed',
        'itch',
        'elevation',
        'changed',
        'has_sewage_system',
        'has_piped_water',
    ]

    questions_list = []

    questions_list.append({
        'key': 'name',
        'type': 'text',
        'text': 'What is the name of the patient?'
    })
    

    col_data = {col: {'dtype': df[col].dtype, 'unique': df[col].dropna().unique().tolist()} for col in columns_to_analyze}


    for col in desired_order:
        if col == 'name':
            continue
        if col not in col_data:
            continue

        question_data = {'key': col}
        dtype = col_data[col]['dtype']

        if dtype in ['int64', 'float64']:
            min_val = df[col].min()
            max_val = df[col].max()
            question_data['type'] = 'numerical'
            question_data['text'] = f'What is the value for {col}?'
            question_data['min'] = min_val
            question_data['max'] = max_val
            if col == 'diameter_1':
                question_data['text'] = 'What is the largest diameter of the lesion?'
            elif col == 'diameter_2':
                question_data['text'] = 'What is the smallest diameter of the lesion?'
            elif col == 'age':
                question_data['text'] = 'What is the age of the patient?'
            elif col == 'race':
                # Handle the race question separately to be categorical
                question_data['type'] = 'categorical'
                question_data['options'] = ['White', 'Black', 'Hispanic', 'Asian or Pacific Islander', 'Native American', 'Other']
                question_data['text'] = 'What is your race?'

        else:
            unique_values = col_data[col]['unique']

            if 'UNK' in unique_values:
                unique_values.remove('UNK')

            formatted_options = []
            for val in unique_values:
                if isinstance(val, bool):
                    formatted_options.append(str(val).capitalize())
                elif isinstance(val, str):
                    formatted_options.append(val.capitalize())
                else:
                    formatted_options.append(str(int(val)))
            

            if col == 'gender':
                formatted_options = ['Male', 'Female']
            elif 'True' in formatted_options and 'False' in formatted_options:
                formatted_options = ['True', 'False']
            else:
                formatted_options.sort()


            question_data['type'] = 'categorical'
            question_data['options'] = formatted_options
            question_data['text'] = f'What is the {col} of the patient?'

            custom_questions = {
                'gender': 'What is the gender of the patient?',
                'race': 'What is your race?',
                'smoke': 'Does the patient smoke?',
                'drink': 'Does the patient drink alcohol?',
                'skin_cancer_history': 'Does the patient have a history of skin cancer?',
                'cancer_history': 'Does the patient have a history of any other type of cancer?',
                'pesticide': 'Has the patient been exposed to pesticides?',
                'biopsed': "Has the patient's lesion been biopsied?",
                'region': 'What part of the body is the lesion on?',
                'grew': 'Has the lesion grown?',
                'hurt': 'Does the lesion hurt?',
                'bleed': 'Does the lesion bleed?',
                'itch': 'Does the lesion itch?',
                'elevation': 'Is the lesion elevated?',
                'changed': 'Has the lesion changed in appearance?',
                'has_sewage_system': "Does the patient's residence have a sewage system?",
                'has_piped_water': "Does the patient's residence have piped water?",
                'race': "What is the race of the patient?"
            }
            if col in custom_questions:
                question_data['text'] = custom_questions[col]
            
        questions_list.append(question_data)

    return df, questions_list

st.markdown("""
<div class="prediction-box">
    <h1 style="color: #ffffff;">🩺 Disease Prediction System</h1>
    <p style="color: #ffffff;">Please upload your medical image and answer the following questions.</p>
</div>
""", unsafe_allow_html=True)


df, questions = load_data()

if df is None or questions is None:
    st.stop()

if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'answers' not in st.session_state:
    st.session_state.answers = {}

def next_step():
    """Move to the next question/step."""
    st.session_state.current_question_index += 1

def previous_step():
    """Move to the previous question/step."""
    if st.session_state.current_question_index > 0:
        st.session_state.current_question_index -= 1

if st.session_state.current_question_index < len(questions):
    st.header("Upload an Image")

    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0

    uploaded_file = st.file_uploader(
        "Choose a skin lesion image...",
        type=["png", "jpg", "jpeg"],
        key=st.session_state.file_uploader_key
    )
    if uploaded_file is not None:
        st.session_state.answers['uploaded_image'] = uploaded_file
        st.image(uploaded_file, caption="Uploaded Image")
        
    current_question = questions[st.session_state.current_question_index]
    progress = (st.session_state.current_question_index + 1) / len(questions)
    st.progress(progress, text=f"Progress: {st.session_state.current_question_index + 1}/{len(questions)}")

    st.header(current_question['text'])
    
    answer_selected = False
    
    if current_question['type'] == 'categorical':
        options = current_question['options']
        
        previous_answer = st.session_state.answers.get(current_question['key'])
        try:
            default_index = options.index(previous_answer)
        except (ValueError, TypeError):
            default_index = None
            
        st.markdown("<p style='font-size:1rem; margin:0; padding:0;'>Select an option:</p>", unsafe_allow_html=True)
        answer = st.radio(
            "",
            options,
            index=default_index,
            key=current_question['key']
        )
        st.session_state.answers[current_question['key']] = answer
        if answer is not None:
            answer_selected = True

    elif current_question['type'] == 'numerical':
        if current_question['key'] == 'age':
            answer = st.slider(
                "Select a value:",
                min_value=1,
                max_value=100,
                value=int(st.session_state.answers.get(current_question['key'], 1)),
                key=current_question['key']
            )
            st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
        else:
            answer = st.number_input(
                "Enter a value:",
                min_value=current_question['min'],
                max_value=current_question['max'],
                value=st.session_state.answers.get(current_question['key'], current_question['min']),
                key=current_question['key']
            )
        st.session_state.answers[current_question['key']] = answer
        answer_selected = True
    
    elif current_question['type'] == 'text':
        answer = st.text_input(
            "Enter your name:",
            value=st.session_state.answers.get(current_question['key'], ""),
            key=current_question['key']
        )
        st.session_state.answers[current_question['key']] = answer
        if answer:
            answer_selected = True

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.current_question_index < len(questions) - 1:
            st.button("Next", on_click=next_step, disabled=not answer_selected)
        else:
            submit_disabled = not answer_selected or 'uploaded_image' not in st.session_state.answers
            if st.button("Submit", on_click=next_step, disabled=submit_disabled):
                st.session_state.submitted = True
                st.session_state.current_question_index = len(questions)
                st.rerun()

    with col2:
        if st.session_state.current_question_index > 0:
            st.button("Previous", on_click=previous_step)

else:
    st.markdown("<h1>Thank you for completing the questionnaire!</h1>", unsafe_allow_html=True)
    
    patient_name = st.session_state.answers.get('name', 'N/A')
    age = st.session_state.answers.get('age', 'N/A')
    gender = st.session_state.answers.get('gender', 'N/A')
    smoke = st.session_state.answers.get('smoke', 'N/A')
    drink = st.session_state.answers.get('drink', 'N/A')
    skin_cancer_history = st.session_state.answers.get('skin_cancer_history', 'N/A')
    cancer_history = st.session_state.answers.get('cancer_history', 'N/A')
    pesticide = st.session_state.answers.get('pesticide', 'N/A')
    biopsed = st.session_state.answers.get('biopsed', 'N/A')
    region = st.session_state.answers.get('region', 'N/A')
    grew = st.session_state.answers.get('grew', 'N/A')
    hurt = st.session_state.answers.get('hurt', 'N/A')
    bleed = st.session_state.answers.get('bleed', 'N/A')
    itch = st.session_state.answers.get('itch', 'N/A')
    elevation = st.session_state.answers.get('elevation', 'N/A')
    changed = st.session_state.answers.get('changed', 'N/A')
    diameter_1 = st.session_state.answers.get('diameter_1', 'N/A')
    diameter_2 = st.session_state.answers.get('diameter_2', 'N/A')
    has_sewage_system = st.session_state.answers.get('has_sewage_system', 'N/A')
    has_piped_water = st.session_state.answers.get('has_piped_water', 'N/A')
    race = st.session_state.answers.get('race', 'N/A')
    uploaded_file_bytes = st.session_state.answers.get('uploaded_image', None)

    #============== Model ==============#
    # CNN Tf
    image_bytes = uploaded_file_bytes.getvalue()
    img = tf.image.decode_image(image_bytes, channels = 3, expand_animations=False)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.resize(img, (128, 128))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    cnn_output = cnn_tf_model.predict(img)

    # Ensemble
    def True_False(value):
        return 1 if value == "True" else 0

    race_val = 0
    if(race == "White"): race_val = 1
    elif(race == "Black"): race_val = 2
    elif(race == "Hispanic"): race_val = 3
    elif(race == "Asian or Pacific Islander"): race_val = 4
    elif(race == "Native American"): race_val = 5
    elif(race == "Other"): race_val = 6

    scaled_values = sc.transform([[age, diameter_1, diameter_2, race_val]])
    age_s, diameter_1_s, diameter_2_s, race_s = scaled_values[0]
    gender_a = 1 if(gender == "Male") else 0
    smoke_a = True_False(smoke)
    drink_a = True_False(drink)
    skin_cancer_history_a = True_False(skin_cancer_history)
    cancer_history_a = True_False(cancer_history)
    pesticide_a = True_False(pesticide)
    biopsed_a = True_False(biopsed)
    grew_a = True_False(grew)
    hurt_a = True_False(hurt)
    bleed_a = True_False(bleed)
    itch_a = True_False(itch)
    elevation_a = True_False(elevation)
    changed_a = True_False(changed)
    has_sewage_system_a = True_False(has_sewage_system)
    has_piped_water_a = True_False(has_piped_water)

    region_a = []
    if(region == "Abdomen"): region_a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif(region == "Arm"): region_a = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif(region == "Back"): region_a = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif(region == "Chest"): region_a = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif(region == "Ear"): region_a = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif(region == "Face"): region_a = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif(region == "Foot"): region_a = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif(region == "Forearm"): region_a = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif(region == "Hand"): region_a = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif(region == "Lip"): region_a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif(region == "Neck"): region_a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif(region == "Nose"): region_a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif(region == "Scalp"): region_a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif(region == "Thigh"): region_a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Ensemble
    metaData = np.array([[age_s, gender_a, smoke_a, drink_a, skin_cancer_history_a, cancer_history_a, pesticide_a,
                 biopsed_a, grew_a, hurt_a, bleed_a, itch_a, elevation_a, changed_a, diameter_1_s, diameter_2_s,
                 has_sewage_system_a, has_piped_water_a, race_s, *region_a]])

    ensemble_input = np.concat([cnn_output, metaData], axis = 1)
    ensemble_output = ensemble_clf.predict(ensemble_input)

    disease = orderLabels[ensemble_output][0]

    predc = np.array([
        xgb_clf.predict(ensemble_input),
        lgb_clf.predict(ensemble_input)
    ])

    (unique, counts) = np.unique(predc, return_counts = True)
    frequency = dict(zip(unique, counts))
    pred_class = max(frequency, key = frequency.get)

    percentage = round(frequency[pred_class] / predc.size * 100)
    # Ensemble
    #============== Model ==============#

    image_src = "https://www.pngitem.com/pimgs/m/150-1503945_anonymous-user-default-user-icon-hd-png-download.png"

    st.markdown(f"""
        <div class="final-card">
            <div class="final-card-image" style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <div style="width: 150px; height: 150px; margin-bottom: 1rem;">
                    <img src="{image_src}" class="uploaded-image-circle">
                </div>
                <hr style="width: 100%; border-top: 1px solid #444; margin: 1rem 0;">
                <div class="percentage-circle" style="--percentage: {percentage}%;">
                    <div class="percentage-text">{str(percentage)}%</div>
                </div>
            </div>
            <div class="final-card-info">
                <h3 style="flex:1 1 100%;">Patient Information</h3>
                <div class="info-row">
                    <h4 class="info-item" style = "text-align: center"><strong>Name:</strong> {patient_name}</h4>
                </div>
                <div class="info-row">
                    <p class="info-item" style = "margin: 0"><strong>Age:</strong> {age}</p>
                    <p class="info-item" style = "margin: 0"><strong>Gender:</strong> {gender}</p>
                </div>
                <div class="info-row">
                    <p class="info-item" style = "margin: 0"><strong>Race:</strong> {race}</p>
                    <p class="info-item" style = "margin: 0"><strong>Region:</strong> {region}</p>
                </div>
                <div class="info-row">
                    <p class="info-item" style = "margin: 0"><strong>Smokes:</strong> {smoke}</p>
                    <p class="info-item" style = "margin: 0"><strong>Skin Cancer History:</strong> {skin_cancer_history}</p>
                </div>
                <div class="info-row">
                    <p class="info-item" style = "margin: 0"><strong>Cancer History:</strong> {cancer_history}</p>
                    <p class="info-item" style = "margin: 0"><strong>Diameter 1:</strong> {diameter_1}</p>
                </div>
                <div class="info-row">
                    <p class="info-item" style = "margin: 0"><strong>Diameter 2:</strong> {diameter_2}</p>
                </div>
                <h3 style="flex:1 1 100%;">Predicted Disease: {disease}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    if st.button("Start a New Examination"):
        st.session_state.clear()
        st.rerun()