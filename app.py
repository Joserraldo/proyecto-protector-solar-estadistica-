import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Cargar modelo entrenado
# ==========================================
try:
    model = joblib.load("modelo_protector_solar_mejorado.pkl")
    EXPECTED_FEATURES = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [
        'Edad',
        'Género_Femenino', 'Género_Masculino', 'Género_Prefiero no decirlo',
        'Tipo_de_piel_Grasa', 'Tipo_de_piel_Mixta', 'Tipo_de_piel_Normal', 'Tipo_de_piel_Seca', 'Tipo_de_piel_Sensible',
        'Color_de_piel_Clara', 'Color_de_piel_Morena clara', 'Color_de_piel_Morena oscura', 'Color_de_piel_Muy clara', 'Color_de_piel_Oscura',
        'Tu_piel_tiende_a_enrojecerse_o_quemarse_fácilmente_con_el_sol_A veces', 
        'Tu_piel_tiende_a_enrojecerse_o_quemarse_fácilmente_con_el_sol_No', 
        'Tu_piel_tiende_a_enrojecerse_o_quemarse_fácilmente_con_el_sol_Sí',
        'Cuántas_horas_promedio_pasas_al_día_expuesto_al_sol_Entre 1 y 3 horas', 
        'Cuántas_horas_promedio_pasas_al_día_expuesto_al_sol_Más de 3 horas', 
        'Cuántas_horas_promedio_pasas_al_día_expuesto_al_sol_Menos de 1 hora',
        'Qué_tan_frecuente_realizas_actividades_al_aire_libre_Casi nunca', 
        'Qué_tan_frecuente_realizas_actividades_al_aire_libre_Frecuentemente', 
        'Qué_tan_frecuente_realizas_actividades_al_aire_libre_Muy frecuentemente', 
        'Qué_tan_frecuente_realizas_actividades_al_aire_libre_Ocasionalmente'
    ]
except FileNotFoundError:
    st.error("🚨 ERROR: El archivo del modelo 'modelo_protector_solar_mejorado.pkl' no fue encontrado.")
    st.stop()
except Exception as e:
    st.error(f"🚨 ERROR al cargar el modelo: {e}")
    st.stop()

# ==========================================
# Configuración de Página
# ==========================================
st.set_page_config(
    page_title="Recomendador de Protector Solar", 
    page_icon="🧴", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado para mejor visualización
st.markdown("""
    <style>
    .big-emoji {
        font-size: 3rem;
        text-align: center;
        margin: 10px 0;
    }
    .option-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 5px;
        transition: all 0.3s ease;
    }
    .option-card-selected {
        padding: 15px;
        border-radius: 10px;
        background-color: #ff6b35;
        color: white;
        text-align: center;
        margin: 5px;
        box-shadow: inset 0 4px 8px rgba(0,0,0,0.3);
        transform: scale(0.98);
        border: 3px solid #f7931e;
    }
    .stRadio > label {
        font-weight: 600;
    }
    div[data-testid="stButton"] button {
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# Cabecera Principal
# ==========================================
col_header1, col_header2, col_header3 = st.columns([1, 2, 1])

with col_header2:
    st.markdown("<h1 style='text-align: center;'>☀️ Sunscreen AI Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #ff6b35;'>Tu Asistente Personal de Protección Solar</h3>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; font-size: 1.1rem;'>
        Descubre si necesitas usar protector solar diariamente basado en tu perfil único. 
        ¡Cuida tu piel con inteligencia artificial! 🧠✨
        </p>
    """, unsafe_allow_html=True)

st.divider()

# ==========================================
# Sección 1: Datos Personales
# ==========================================
with st.container(border=True):
    st.markdown("### 👤 Paso 1: Cuéntanos sobre ti")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎂 Tu Edad")
        edad = st.slider("", min_value=10, max_value=100, step=1, value=30, label_visibility="collapsed")
        st.caption(f"Tienes {edad} años")
    
    with col2:
        st.markdown("#### 🚻 Tu Género")
        
        # Opciones con emojis
        genero_options = {
            "👩 Femenino": "Femenino",
            "👨 Masculino": "Masculino",
            "🧑 Prefiero no decirlo": "Prefiero no decirlo"
        }
        
        genero_display = st.radio(
            "",
            options=list(genero_options.keys()),
            horizontal=True,
            label_visibility="collapsed"
        )
        genero = genero_options[genero_display]

st.divider()

# ==========================================
# Sección 2: Características de la Piel
# ==========================================
with st.container(border=True):
    st.markdown("### 🧴 Paso 2: Características de tu piel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💧 Tipo de Piel")
        
        tipo_piel_options = {
            "✨ Normal": "Normal",
            "💦 Grasa": "Grasa",
            "🏜️ Seca": "Seca",
            "🔀 Mixta": "Mixta",
            "🌸 Sensible": "Sensible"
        }
        
        tipo_piel_display = st.selectbox(
            "",
            options=list(tipo_piel_options.keys()),
            label_visibility="collapsed"
        )
        tipo_piel = tipo_piel_options[tipo_piel_display]
        
        # Descripción del tipo de piel
        descripciones_tipo = {
            "Normal": "Piel equilibrada, sin exceso de grasa ni sequedad",
            "Grasa": "Piel con brillo, poros dilatados",
            "Seca": "Piel tirante, escamosa o áspera",
            "Mixta": "Grasa en zona T, seca en mejillas",
            "Sensible": "Piel reactiva, se irrita fácilmente"
        }
        st.caption(f"ℹ️ {descripciones_tipo[tipo_piel]}")
    
    with col2:
        st.markdown("#### 🎨 Color de Piel (Escala Fitzpatrick)")
        
        # Representación visual de colores de piel
        color_piel_options = {
            "🤍 Muy clara (Tipo I-II)": "Muy clara",
            "🧑🏻 Clara (Tipo III)": "Clara",
            "🧑🏽 Morena clara (Tipo IV)": "Morena clara",
            "🧑🏾 Morena oscura (Tipo V)": "Morena oscura",
            "🧑🏿 Oscura (Tipo VI)": "Oscura"
        }
        
        color_piel_display = st.selectbox(
            "",
            options=list(color_piel_options.keys()),
            label_visibility="collapsed"
        )
        color_piel = color_piel_options[color_piel_display]
        
        # Información sobre fototipos
        fototipo_info = {
            "Muy clara": "Siempre se quema, nunca se broncea",
            "Clara": "Generalmente se quema, broncea mínimamente",
            "Morena clara": "A veces se quema, broncea gradualmente",
            "Morena oscura": "Rara vez se quema, broncea fácilmente",
            "Oscura": "Muy rara vez se quema, broncea intensamente"
        }
        st.caption(f"ℹ️ {fototipo_info[color_piel]}")

st.divider()

# ==========================================
# Sección 3: Sensibilidad y Exposición Solar
# ==========================================
with st.container(border=True):
    st.markdown("### ☀️ Paso 3: Tu exposición al sol")
    
    # Sensibilidad a quemaduras
    st.markdown("#### 🔥 ¿Tu piel se enrojece o quema fácilmente con el sol?")
    
    # Valor por defecto
    if 'quemarse' not in st.session_state:
        st.session_state.quemarse = "A veces"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        btn_type = "primary" if st.session_state.quemarse == "Sí" else "secondary"
        if st.button("🚨 Sí, muy fácil", use_container_width=True, key="quema_si", type=btn_type):
            st.session_state.quemarse = "Sí"
    
    with col2:
        btn_type = "primary" if st.session_state.quemarse == "A veces" else "secondary"
        if st.button("⚠️ A veces", use_container_width=True, key="quema_aveces", type=btn_type):
            st.session_state.quemarse = "A veces"
    
    with col3:
        btn_type = "primary" if st.session_state.quemarse == "No" else "secondary"
        if st.button("✅ No, raramente", use_container_width=True, key="quema_no", type=btn_type):
            st.session_state.quemarse = "No"
    
    quemarse = st.session_state.quemarse
    
    st.markdown("---")
    
    # Horas de exposición
    st.markdown("#### ⏰ ¿Cuántas horas al día estás expuesto al sol?")
    
    # Valor por defecto
    if 'horas_sol' not in st.session_state:
        st.session_state.horas_sol = "Entre 1 y 3 horas"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        card_class = 'option-card-selected' if st.session_state.horas_sol == "Menos de 1 hora" else 'option-card'
        st.markdown(f"""
            <div class='{card_class}'>
                <div class='big-emoji'>🌤️</div>
                <strong>Menos de 1 hora</strong><br>
                <small>Interior la mayor parte del día</small>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Seleccionar", key="hora1", use_container_width=True):
            st.session_state.horas_sol = "Menos de 1 hora"
            st.rerun()
    
    with col2:
        card_class = 'option-card-selected' if st.session_state.horas_sol == "Entre 1 y 3 horas" else 'option-card'
        st.markdown(f"""
            <div class='{card_class}'>
                <div class='big-emoji'>☀️</div>
                <strong>Entre 1 y 3 horas</strong><br>
                <small>Exposición moderada</small>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Seleccionar", key="hora2", use_container_width=True):
            st.session_state.horas_sol = "Entre 1 y 3 horas"
            st.rerun()
    
    with col3:
        card_class = 'option-card-selected' if st.session_state.horas_sol == "Más de 3 horas" else 'option-card'
        st.markdown(f"""
            <div class='{card_class}'>
                <div class='big-emoji'>🌞</div>
                <strong>Más de 3 horas</strong><br>
                <small>Exposición intensa</small>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Seleccionar", key="hora3", use_container_width=True):
            st.session_state.horas_sol = "Más de 3 horas"
            st.rerun()
    
    horas_sol = st.session_state.horas_sol
    
    st.markdown("---")
    
    # Actividades al aire libre
    st.markdown("#### 🏃 ¿Con qué frecuencia realizas actividades al aire libre?")
    
    # Valor por defecto
    if 'aire_libre' not in st.session_state:
        st.session_state.aire_libre = "Ocasionalmente"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        card_class = 'option-card-selected' if st.session_state.aire_libre == "Casi nunca" else 'option-card'
        st.markdown(f"""
            <div class='{card_class}'>
                <div class='big-emoji'>🏠</div>
                <strong>Casi nunca</strong>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Seleccionar", key="aire1", use_container_width=True):
            st.session_state.aire_libre = "Casi nunca"
            st.rerun()
    
    with col2:
        card_class = 'option-card-selected' if st.session_state.aire_libre == "Ocasionalmente" else 'option-card'
        st.markdown(f"""
            <div class='{card_class}'>
                <div class='big-emoji'>🚶</div>
                <strong>Ocasionalmente</strong>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Seleccionar", key="aire2", use_container_width=True):
            st.session_state.aire_libre = "Ocasionalmente"
            st.rerun()
    
    with col3:
        card_class = 'option-card-selected' if st.session_state.aire_libre == "Frecuentemente" else 'option-card'
        st.markdown(f"""
            <div class='{card_class}'>
                <div class='big-emoji'>🏃</div>
                <strong>Frecuentemente</strong>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Seleccionar", key="aire3", use_container_width=True):
            st.session_state.aire_libre = "Frecuentemente"
            st.rerun()
    
    with col4:
        card_class = 'option-card-selected' if st.session_state.aire_libre == "Muy frecuentemente" else 'option-card'
        st.markdown(f"""
            <div class='{card_class}'>
                <div class='big-emoji'>⛰️</div>
                <strong>Muy frecuente</strong>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Seleccionar", key="aire4", use_container_width=True):
            st.session_state.aire_libre = "Muy frecuentemente"
            st.rerun()
    
    aire_libre = st.session_state.aire_libre

st.divider()

# ==========================================
# Botón de Predicción
# ==========================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button(
        "🔮 OBTENER MI RECOMENDACIÓN PERSONALIZADA",
        use_container_width=True,
        type="primary",
        help="Haz clic para analizar tu perfil y recibir tu recomendación"
    )

st.divider()

# ==========================================
# Lógica de Predicción
# ==========================================

if predict_button:
    with st.spinner("🔄 Analizando tu perfil..."):
        # 1. Convertir entrada a DataFrame
        entrada = pd.DataFrame({
            "Edad": [edad],
            "Género": [genero],
            "Tipo_de_piel": [tipo_piel],
            "Color_de_piel": [color_piel],
            "Tu_piel_tiende_a_enrojecerse_o_quemarse_fácilmente_con_el_sol": [quemarse],
            "Cuántas_horas_promedio_pasas_al_día_expuesto_al_sol": [horas_sol],
            "Qué_tan_frecuente_realizas_actividades_al_aire_libre": [aire_libre]
        })

        # 2. Procesar y alinear columnas
        try:
            entrada_encoded = pd.get_dummies(entrada)
            
            final_features = []
            for col in EXPECTED_FEATURES:
                if col in entrada_encoded.columns:
                    final_features.append(entrada_encoded[col])
                else:
                    final_features.append(pd.Series([0], name=col))
            
            entrada_aligned = pd.concat(final_features, axis=1)
            if 'Edad' in entrada_aligned.columns:
                entrada_aligned['Edad'] = entrada['Edad']
            entrada_aligned = entrada_aligned[EXPECTED_FEATURES]

        except Exception as e:
            st.error(f"Error al procesar la entrada: {e}")
            st.stop()

        # 3. Predicción
        pred = model.predict(entrada_aligned)[0]
        prob_si = model.predict_proba(entrada_aligned)[0][1]

    # ==========================================
    # Mostrar Resultado
    # ==========================================
    
    st.markdown("## 🎯 Tu Recomendación Personalizada")
    
    if pred == 1:
        # Recomendación SÍ
        st.balloons()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
                <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%); 
                border-radius: 20px; color: white;'>
                    <div style='font-size: 5rem;'>🧴</div>
                    <h2>¡USO DIARIO!</h2>
                    <h1 style='font-size: 3rem; margin: 10px 0;'>SÍ</h1>
                    <p style='font-size: 1.2rem;'>Confianza: {:.1f}%</p>
                </div>
            """.format(prob_si*100), unsafe_allow_html=True)
        
        with col2:
            st.success("### ✅ ¡Protección Solar Diaria Altamente Recomendada!", icon="🛡️")
            st.markdown("""
                Tu perfil indica que tu piel necesita **protección constante** contra los rayos UV. 
                Esto significa que deberías aplicar protector solar **todos los días**, incluso en días nublados.
            """)
            
            # Recomendación de SPF según probabilidad
            if prob_si > 0.85:
                st.error("### 🚨 Nivel de Protección: MÁXIMO", icon="⚠️")
                st.markdown("""
                    - **SPF Recomendado:** 50+ (Muy Alto)
                    - **Aplicación:** Cada 2 horas
                    - **Cantidad:** 2mg/cm² (generosa)
                    - **Tipo:** Resistente al agua y de amplio espectro
                    - **Extra:** Usa sombrero, gafas y ropa protectora
                """)
            elif prob_si > 0.65:
                st.warning("### ⚠️ Nivel de Protección: ALTO", icon="☀️")
                st.markdown("""
                    - **SPF Recomendado:** 30-50 (Alto)
                    - **Aplicación:** Por la mañana y re-aplicar si sales
                    - **Cantidad:** Generosa en todas las áreas expuestas
                    - **Tipo:** De amplio espectro
                    - **Extra:** Re-aplicar después de nadar o sudar
                """)
            else:
                st.info("### 📋 Nivel de Protección: MODERADO", icon="🌤️")
                st.markdown("""
                    - **SPF Recomendado:** 15-30 (Moderado)
                    - **Aplicación:** Diaria, especialmente antes de salir
                    - **Cantidad:** Aplicación uniforme
                    - **Tipo:** De amplio espectro
                    - **Extra:** Aumentar SPF en verano o exposición prolongada
                """)
    
    else:
        # Recomendación NO
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
                <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%); 
                border-radius: 20px; color: white;'>
                    <div style='font-size: 5rem;'>☀️</div>
                    <h2>Uso Condicional</h2>
                    <h1 style='font-size: 3rem; margin: 10px 0;'>NO</h1>
                    <p style='font-size: 1.2rem;'>Confianza: {:.1f}%</p>
                </div>
            """.format((1-prob_si)*100), unsafe_allow_html=True)
        
        with col2:
            st.info("### ℹ️ Protección Solar Condicional", icon="🌤️")
            st.markdown("""
                Tu perfil sugiere que el uso diario de protector solar **no es estrictamente necesario**, 
                pero se recomienda durante:
                
                - ☀️ Exposición solar prolongada (>30 minutos)
                - 🏖️ Actividades al aire libre intensas
                - 🌊 Días de playa o piscina
                - ⛰️ Alta montaña o nieve
                - 🌤️ Días muy soleados o en verano
                
                **Consejo:** Ten siempre un protector solar a mano para estas situaciones.
            """)
    
    # Resumen del perfil
    st.markdown("---")
    with st.expander("📊 Ver Resumen de tu Perfil"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Edad", f"{edad} años")
            st.metric("Género", genero)
        
        with col2:
            st.metric("Tipo de Piel", tipo_piel)
            st.metric("Color de Piel", color_piel)
        
        with col3:
            st.metric("Exposición Solar", horas_sol)
            st.metric("Actividades", aire_libre)

# ==========================================
# Pie de página
# ==========================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🧪 <strong>Sunscreen AI Predictor</strong> | Proyecto de Machine Learning y Estadística</p>
        <p>Desarrollado con ❤️ usando Streamlit y Random Forest</p>
        <p><small>⚠️ Esta herramienta es educativa. Consulta a un dermatólogo para recomendaciones personalizadas.</small></p>
    </div>
""", unsafe_allow_html=True)