import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import json
import io

st.set_page_config(
    page_title="Visualizador NACA",
    page_icon="✈️",
    layout="wide"
)

st.title("Visualizador de Perfiles NACA")
st.markdown("Herramienta interactiva para visualizar y comparar perfiles aerodinámicos NACA con diferentes configuraciones.")

def calcular_espesor_naca(x_norm, t):
    """Calcula la distribución de espesor para perfiles NACA 4 dígitos"""
    yt = 5 * t * (
        0.2969 * np.sqrt(x_norm) 
        - 0.1260 * x_norm 
        - 0.3516 * x_norm**2 
        + 0.2843 * x_norm**3 
        - 0.1015 * x_norm**4
    )
    return yt

def calcular_linea_curvatura(x_norm, m, p):
    """Calcula la línea de curvatura media para perfiles NACA 4 dígitos"""
    yc = np.zeros_like(x_norm)
    dyc_dx = np.zeros_like(x_norm)
    
    if m == 0 or p == 0:
        return yc, dyc_dx
    
    mask_front = x_norm <= p
    mask_rear = x_norm > p
    
    yc[mask_front] = (m / p**2) * (2 * p * x_norm[mask_front] - x_norm[mask_front]**2)
    yc[mask_rear] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x_norm[mask_rear] - x_norm[mask_rear]**2)
    
    dyc_dx[mask_front] = (2 * m / p**2) * (p - x_norm[mask_front])
    dyc_dx[mask_rear] = (2 * m / (1 - p)**2) * (p - x_norm[mask_rear])
    
    return yc, dyc_dx

def perfil_naca_4digitos(x, cuerda, m, p, t):
    """Calcula las coordenadas completas del perfil NACA 4 dígitos"""
    x_norm = x / cuerda
    
    yt = calcular_espesor_naca(x_norm, t) * cuerda
    yc, dyc_dx = calcular_linea_curvatura(x_norm, m, p)
    yc = yc * cuerda
    
    theta = np.arctan(dyc_dx)
    
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    return xu, yu, xl, yl, yc, yt

NACA_5_DIGIT_PARAMS = {
    210: {'r': 0.0580, 'k1': 361.400},
    220: {'r': 0.1260, 'k1': 51.640},
    230: {'r': 0.2025, 'k1': 15.957},
    240: {'r': 0.2900, 'k1': 6.643},
    250: {'r': 0.3910, 'k1': 3.230},
}

NACA_5_DIGIT_REFLEX_PARAMS = {
    221: {'r': 0.1300, 'k1': 51.990, 'k2_k1': 0.000764},
    231: {'r': 0.2170, 'k1': 15.793, 'k2_k1': 0.00677},
    241: {'r': 0.3180, 'k1': 6.520, 'k2_k1': 0.0303},
    251: {'r': 0.4410, 'k1': 3.191, 'k2_k1': 0.1355},
}

def calcular_linea_curvatura_5digitos(x_norm, lp_code, is_reflex):
    """Calcula la línea de curvatura media para perfiles NACA 5 dígitos"""
    yc = np.zeros_like(x_norm)
    dyc_dx = np.zeros_like(x_norm)
    
    if is_reflex:
        params = NACA_5_DIGIT_REFLEX_PARAMS
        key = lp_code
        if key not in params:
            key = 231
        r = params[key]['r']
        k1 = params[key]['k1']
        k2_k1 = params[key]['k2_k1']
        
        mask_front = x_norm <= r
        mask_rear = x_norm > r
        
        yc[mask_front] = (k1 / 6) * ((x_norm[mask_front] - r)**3 - k2_k1 * (1 - r)**3 * x_norm[mask_front] - r**3 * x_norm[mask_front] + r**3)
        yc[mask_rear] = (k1 / 6) * (k2_k1 * (x_norm[mask_rear] - r)**3 - k2_k1 * (1 - r)**3 * x_norm[mask_rear] - r**3 * x_norm[mask_rear] + r**3)
        
        dyc_dx[mask_front] = (k1 / 6) * (3 * (x_norm[mask_front] - r)**2 - k2_k1 * (1 - r)**3 - r**3)
        dyc_dx[mask_rear] = (k1 / 6) * (3 * k2_k1 * (x_norm[mask_rear] - r)**2 - k2_k1 * (1 - r)**3 - r**3)
    else:
        params = NACA_5_DIGIT_PARAMS
        key = lp_code
        if key not in params:
            key = 230
        r = params[key]['r']
        k1 = params[key]['k1']
        
        mask_front = x_norm <= r
        mask_rear = x_norm > r
        
        yc[mask_front] = (k1 / 6) * (x_norm[mask_front]**3 - 3 * r * x_norm[mask_front]**2 + r**2 * (3 - r) * x_norm[mask_front])
        yc[mask_rear] = (k1 * r**3 / 6) * (1 - x_norm[mask_rear])
        
        dyc_dx[mask_front] = (k1 / 6) * (3 * x_norm[mask_front]**2 - 6 * r * x_norm[mask_front] + r**2 * (3 - r))
        dyc_dx[mask_rear] = -(k1 * r**3 / 6)
    
    return yc, dyc_dx

def perfil_naca_5digitos(x, cuerda, lp_code, is_reflex, t):
    """Calcula las coordenadas completas del perfil NACA 5 dígitos"""
    x_norm = x / cuerda
    
    yt = calcular_espesor_naca(x_norm, t) * cuerda
    yc, dyc_dx = calcular_linea_curvatura_5digitos(x_norm, lp_code, is_reflex)
    yc = yc * cuerda
    
    theta = np.arctan(dyc_dx)
    
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    return xu, yu, xl, yl, yc, yt

def parsear_naca(codigo):
    """Parsea un código NACA de 4 o 5 dígitos"""
    if not codigo.isdigit():
        return None, None, None, None
    
    if len(codigo) == 4:
        m = int(codigo[0]) / 100.0
        p = int(codigo[1]) / 10.0
        t = int(codigo[2:4]) / 100.0
        return m, p, t, '4digit'
    
    elif len(codigo) == 5:
        lp_code = int(codigo[0:2]) * 10 + int(codigo[2])
        is_reflex = int(codigo[2]) == 1
        t = int(codigo[3:5]) / 100.0
        return lp_code, is_reflex, t, '5digit'
    
    return None, None, None, None

def generar_perfil_naca(x, cuerda, param1, param2, t, tipo):
    """Genera el perfil NACA según el tipo (4 o 5 dígitos)"""
    if tipo == '4digit':
        return perfil_naca_4digitos(x, cuerda, param1, param2, t)
    else:
        return perfil_naca_5digitos(x, cuerda, param1, param2, t)

def calcular_propiedades(x, yt, yc, cuerda):
    """Calcula propiedades aerodinámicas del perfil"""
    espesor_max = float(np.max(yt) * 2)
    pos_espesor_max = float(x[np.argmax(yt)])
    
    curvatura_max = float(np.max(np.abs(yc)))
    if curvatura_max > 0:
        pos_curvatura_max = float(x[np.argmax(np.abs(yc))])
    else:
        pos_curvatura_max = 0.0
    
    area = float(np.trapezoid(yt * 2, x))
    
    return {
        'espesor_max': espesor_max,
        'pos_espesor_max': pos_espesor_max,
        'espesor_max_pct': float((espesor_max / cuerda) * 100),
        'pos_espesor_max_pct': float((pos_espesor_max / cuerda) * 100),
        'curvatura_max': curvatura_max,
        'pos_curvatura_max': pos_curvatura_max,
        'curvatura_max_pct': float((curvatura_max / cuerda) * 100) if cuerda > 0 else 0.0,
        'pos_curvatura_max_pct': float((pos_curvatura_max / cuerda) * 100) if cuerda > 0 else 0.0,
        'area': area
    }

def calcular_reynolds(cuerda_mm, velocidad, densidad, viscosidad):
    """Calcula el número de Reynolds
    
    Args:
        cuerda_mm: Longitud de cuerda en mm
        velocidad: Velocidad del flujo en m/s
        densidad: Densidad del aire en kg/m³
        viscosidad: Viscosidad dinámica en Pa·s
    
    Returns:
        Número de Reynolds (adimensional)
    """
    cuerda_m = cuerda_mm / 1000.0
    reynolds = (densidad * velocidad * cuerda_m) / viscosidad
    return float(reynolds)

def calcular_alpha_zero_lift(x_norm, yc_norm):
    """Calcula el ángulo de sustentación cero (α_L0) usando teoría de perfil delgado
    
    Para perfiles NACA, α_L0 se aproxima por la integral de la pendiente de curvatura
    """
    if np.max(np.abs(yc_norm)) < 1e-6:
        return 0.0
    
    dyc_dx = np.gradient(yc_norm, x_norm)
    theta = np.arccos(1 - 2 * x_norm)
    integrand = dyc_dx * (np.cos(theta) - 1)
    alpha_L0 = -(1 / np.pi) * np.trapezoid(integrand, theta)
    
    return float(np.degrees(alpha_L0))

def calcular_aerodinamica_avanzada(x, yc, cuerda, velocidad, densidad, viscosidad, 
                                    aspect_ratio, oswald_e, cd0):
    """Calcula propiedades aerodinámicas avanzadas
    
    Args:
        x: Coordenadas x del perfil
        yc: Línea de curvatura media
        cuerda: Longitud de cuerda en mm
        velocidad: Velocidad del flujo en m/s
        densidad: Densidad del aire en kg/m³
        viscosidad: Viscosidad dinámica en Pa·s
        aspect_ratio: Relación de aspecto del ala
        oswald_e: Factor de eficiencia de Oswald
        cd0: Coeficiente de arrastre parásito
    
    Returns:
        Diccionario con Reynolds, α_L0, α_opt, CL_opt, L/D_max
    """
    reynolds = calcular_reynolds(cuerda, velocidad, densidad, viscosidad)
    
    x_norm = x / cuerda
    yc_norm = yc / cuerda
    alpha_L0 = calcular_alpha_zero_lift(x_norm, yc_norm)
    
    cl_alpha_2d = 2 * np.pi
    cl_alpha = cl_alpha_2d * aspect_ratio / (2 + np.sqrt(4 + aspect_ratio**2))
    
    k = 1 / (np.pi * aspect_ratio * oswald_e)
    
    cl_opt = np.sqrt(cd0 / k)
    
    alpha_opt_rad = (cl_opt / cl_alpha) + np.radians(alpha_L0)
    alpha_opt = float(np.degrees(alpha_opt_rad))
    
    ld_max = cl_opt / (cd0 + k * cl_opt**2)
    
    return {
        'reynolds': reynolds,
        'alpha_L0': alpha_L0,
        'alpha_opt': alpha_opt,
        'cl_opt': float(cl_opt),
        'cl_alpha': float(np.degrees(cl_alpha)),
        'ld_max': float(ld_max),
        'k_induced': float(k)
    }

st.sidebar.header("Configuración")

modo = st.sidebar.selectbox(
    "Modo de visualización",
    ["Perfiles NACA", "Visualización 3D", "Importar Perfil"]
)

if modo == "Perfiles NACA":
    st.sidebar.subheader("Perfil A")
    naca_a = st.sidebar.text_input("Código NACA A", value="0015", max_chars=5)
    cuerda_a = st.sidebar.slider("Cuerda A (mm)", 20, 200, 80, 5)
    
    st.sidebar.subheader("Perfil B")
    naca_b = st.sidebar.text_input("Código NACA B", value="23012", max_chars=5)
    cuerda_b = st.sidebar.slider("Cuerda B (mm)", 20, 200, 90, 5)
    
    num_puntos = st.sidebar.slider("Número de puntos", 50, 500, 200, 50)
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("Condiciones de Operación", expanded=True):
        velocidad = st.number_input("Velocidad (m/s)", min_value=1.0, max_value=300.0, value=30.0, step=5.0)
        
        usar_isa = st.checkbox("Usar atmósfera estándar (ISA)", value=True)
        if usar_isa:
            densidad = 1.225
            viscosidad = 1.81e-5
            st.caption(f"ρ = {densidad} kg/m³, μ = {viscosidad:.2e} Pa·s")
        else:
            densidad = st.number_input("Densidad (kg/m³)", min_value=0.1, max_value=2.0, value=1.225, step=0.01, format="%.3f")
            viscosidad = st.number_input("Viscosidad (Pa·s)", min_value=1e-6, max_value=1e-4, value=1.81e-5, step=1e-6, format="%.2e")
        
        st.markdown("**Parámetros del Ala**")
        aspect_ratio = st.number_input("Relación de aspecto (AR)", min_value=1.0, max_value=30.0, value=8.0, step=0.5)
        oswald_e = st.number_input("Factor de Oswald (e)", min_value=0.5, max_value=1.0, value=0.85, step=0.05)
        cd0 = st.number_input("CD₀ (arrastre parásito)", min_value=0.005, max_value=0.1, value=0.02, step=0.005, format="%.3f")
    
    param1_a, param2_a, t_a, tipo_a = parsear_naca(naca_a)
    param1_b, param2_b, t_b, tipo_b = parsear_naca(naca_b)
    
    if param1_a is None:
        st.error(f"Código NACA '{naca_a}' inválido. Use 4 o 5 dígitos (ej: 0015, 2412, 23012)")
        st.stop()
    if param1_b is None:
        st.error(f"Código NACA '{naca_b}' inválido. Use 4 o 5 dígitos (ej: 0015, 2412, 23012)")
        st.stop()
    
    x_a = np.linspace(0, cuerda_a, num_puntos)
    x_b = np.linspace(0, cuerda_b, num_puntos)
    
    xu_a, yu_a, xl_a, yl_a, yc_a, yt_a = generar_perfil_naca(x_a, cuerda_a, param1_a, param2_a, t_a, tipo_a)
    xu_b, yu_b, xl_b, yl_b, yc_b, yt_b = generar_perfil_naca(x_b, cuerda_b, param1_b, param2_b, t_b, tipo_b)
    
    props_a = calcular_propiedades(x_a, yt_a, yc_a, cuerda_a)
    props_b = calcular_propiedades(x_b, yt_b, yc_b, cuerda_b)
    
    aero_a = calcular_aerodinamica_avanzada(x_a, yc_a, cuerda_a, velocidad, densidad, viscosidad, aspect_ratio, oswald_e, cd0)
    aero_b = calcular_aerodinamica_avanzada(x_b, yc_b, cuerda_b, velocidad, densidad, viscosidad, aspect_ratio, oswald_e, cd0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Perfil NACA {naca_a} - Cuerda {cuerda_a} mm")
        
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        ax1.plot(xu_a, yu_a, 'b', linewidth=2, label='Extradós')
        ax1.plot(xl_a, yl_a, 'r', linewidth=2, label='Intradós')
        if np.max(np.abs(yc_a)) > 0.01:
            ax1.plot(x_a, yc_a, 'g--', linewidth=1.5, label='Línea de curvatura')
        ax1.fill_between(xu_a, yu_a, np.interp(xu_a, xl_a, yl_a), alpha=0.1, color='gray')
        ax1.set_xlabel('x (mm)', fontsize=12)
        ax1.set_ylabel('y (mm)', fontsize=12)
        ax1.set_title(f'Perfil NACA {naca_a} - Cuerda {cuerda_a} mm', fontsize=14)
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader(f"Perfil NACA {naca_b} - Cuerda {cuerda_b} mm")
        
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        ax2.plot(xu_b, yu_b, 'b', linewidth=2, label='Extradós')
        ax2.plot(xl_b, yl_b, 'r', linewidth=2, label='Intradós')
        if np.max(np.abs(yc_b)) > 0.01:
            ax2.plot(x_b, yc_b, 'g--', linewidth=1.5, label='Línea de curvatura')
        ax2.fill_between(xu_b, yu_b, np.interp(xu_b, xl_b, yl_b), alpha=0.1, color='gray')
        ax2.set_xlabel('x (mm)', fontsize=12)
        ax2.set_ylabel('y (mm)', fontsize=12)
        ax2.set_title(f'Perfil NACA {naca_b} - Cuerda {cuerda_b} mm', fontsize=14)
        ax2.set_aspect('equal', adjustable='box')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    
    st.markdown("---")
    st.subheader("Propiedades Aerodinámicas")
    
    prop_col1, prop_col2 = st.columns(2)
    
    with prop_col1:
        st.markdown(f"**NACA {naca_a} (Cuerda: {cuerda_a} mm)**")
        st.write(f"- Espesor máximo: {props_a['espesor_max']:.2f} mm ({props_a['espesor_max_pct']:.1f}%)")
        st.write(f"- Posición espesor máx: {props_a['pos_espesor_max']:.2f} mm ({props_a['pos_espesor_max_pct']:.1f}%)")
        st.write(f"- Curvatura máxima: {props_a['curvatura_max']:.2f} mm ({props_a['curvatura_max_pct']:.1f}%)")
        st.write(f"- Área del perfil: {props_a['area']:.2f} mm²")
    
    with prop_col2:
        st.markdown(f"**NACA {naca_b} (Cuerda: {cuerda_b} mm)**")
        st.write(f"- Espesor máximo: {props_b['espesor_max']:.2f} mm ({props_b['espesor_max_pct']:.1f}%)")
        st.write(f"- Posición espesor máx: {props_b['pos_espesor_max']:.2f} mm ({props_b['pos_espesor_max_pct']:.1f}%)")
        st.write(f"- Curvatura máxima: {props_b['curvatura_max']:.2f} mm ({props_b['curvatura_max_pct']:.1f}%)")
        st.write(f"- Área del perfil: {props_b['area']:.2f} mm²")
    
    st.markdown("---")
    st.subheader("Análisis Aerodinámico Avanzado")
    st.caption(f"Condiciones: V = {velocidad} m/s, AR = {aspect_ratio}, e = {oswald_e}, CD₀ = {cd0}")
    
    aero_col1, aero_col2 = st.columns(2)
    
    with aero_col1:
        st.markdown(f"**NACA {naca_a}**")
        st.write(f"- Número de Reynolds: {aero_a['reynolds']:.2e}")
        st.write(f"- Ángulo sustentación cero (α_L0): {aero_a['alpha_L0']:.2f}°")
        st.write(f"- Ángulo de ataque óptimo (α_opt): {aero_a['alpha_opt']:.2f}°")
        st.write(f"- CL óptimo: {aero_a['cl_opt']:.3f}")
        st.write(f"- Eficiencia máxima (L/D)_max: {aero_a['ld_max']:.2f}")
    
    with aero_col2:
        st.markdown(f"**NACA {naca_b}**")
        st.write(f"- Número de Reynolds: {aero_b['reynolds']:.2e}")
        st.write(f"- Ángulo sustentación cero (α_L0): {aero_b['alpha_L0']:.2f}°")
        st.write(f"- Ángulo de ataque óptimo (α_opt): {aero_b['alpha_opt']:.2f}°")
        st.write(f"- CL óptimo: {aero_b['cl_opt']:.3f}")
        st.write(f"- Eficiencia máxima (L/D)_max: {aero_b['ld_max']:.2f}")
    
    st.markdown("---")
    st.subheader("Comparación de Perfiles")
    
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    ax3.plot(xu_a, yu_a, 'b', linewidth=2, label=f'NACA {naca_a} Extradós')
    ax3.plot(xl_a, yl_a, 'b--', linewidth=2, label=f'NACA {naca_a} Intradós')
    ax3.plot(xu_b, yu_b, 'r', linewidth=2, label=f'NACA {naca_b} Extradós')
    ax3.plot(xl_b, yl_b, 'r--', linewidth=2, label=f'NACA {naca_b} Intradós')
    ax3.set_xlabel('x (mm)', fontsize=12)
    ax3.set_ylabel('y (mm)', fontsize=12)
    ax3.set_title('Comparación de Perfiles NACA', fontsize=14)
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper right')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
    
    st.markdown("---")
    st.subheader("Distribución de Espesor")
    
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(x_a / cuerda_a * 100, yt_a * 2 / cuerda_a * 100, 'b', linewidth=2, label=f'NACA {naca_a}')
    ax4.plot(x_b / cuerda_b * 100, yt_b * 2 / cuerda_b * 100, 'r', linewidth=2, label=f'NACA {naca_b}')
    ax4.set_xlabel('Posición (% cuerda)', fontsize=12)
    ax4.set_ylabel('Espesor (% cuerda)', fontsize=12)
    ax4.set_title('Distribución de Espesor', fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(loc='upper right')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)
    
    st.markdown("---")
    st.subheader("Datos y Exportación")
    
    tab1, tab2, tab3 = st.tabs([f"NACA {naca_a}", f"NACA {naca_b}", "Exportar"])
    
    with tab1:
        df_a = pd.DataFrame({
            'x (mm)': x_a,
            'xu (mm)': xu_a,
            'yu (mm)': yu_a,
            'xl (mm)': xl_a,
            'yl (mm)': yl_a,
            'yc (mm)': yc_a,
            'espesor (mm)': yt_a * 2
        })
        st.dataframe(df_a.round(4), width='stretch', height=300)
    
    with tab2:
        df_b = pd.DataFrame({
            'x (mm)': x_b,
            'xu (mm)': xu_b,
            'yu (mm)': yu_b,
            'xl (mm)': xl_b,
            'yl (mm)': yl_b,
            'yc (mm)': yc_b,
            'espesor (mm)': yt_b * 2
        })
        st.dataframe(df_b.round(4), width='stretch', height=300)
    
    with tab3:
        st.markdown("### Exportar Coordenadas")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.markdown(f"**NACA {naca_a}**")
            
            csv_a = df_a.round(6).to_csv(index=False)
            st.download_button(
                label="Descargar CSV",
                data=csv_a,
                file_name=f"NACA_{naca_a}_{cuerda_a}mm.csv",
                mime="text/csv"
            )
            
            json_data_a = {
                'perfil': f'NACA {naca_a}',
                'cuerda_mm': cuerda_a,
                'propiedades': props_a,
                'aerodinamica': aero_a,
                'condiciones_operacion': {
                    'velocidad_ms': velocidad,
                    'densidad_kg_m3': densidad,
                    'viscosidad_Pa_s': viscosidad,
                    'aspect_ratio': aspect_ratio,
                    'oswald_e': oswald_e,
                    'cd0': cd0
                },
                'coordenadas': {
                    'x': x_a.round(6).tolist(),
                    'xu': xu_a.round(6).tolist(),
                    'yu': yu_a.round(6).tolist(),
                    'xl': xl_a.round(6).tolist(),
                    'yl': yl_a.round(6).tolist()
                }
            }
            st.download_button(
                label="Descargar JSON",
                data=json.dumps(json_data_a, indent=2),
                file_name=f"NACA_{naca_a}_{cuerda_a}mm.json",
                mime="application/json"
            )
        
        with export_col2:
            st.markdown(f"**NACA {naca_b}**")
            
            csv_b = df_b.round(6).to_csv(index=False)
            st.download_button(
                label="Descargar CSV",
                data=csv_b,
                file_name=f"NACA_{naca_b}_{cuerda_b}mm.csv",
                mime="text/csv"
            )
            
            json_data_b = {
                'perfil': f'NACA {naca_b}',
                'cuerda_mm': cuerda_b,
                'propiedades': props_b,
                'aerodinamica': aero_b,
                'condiciones_operacion': {
                    'velocidad_ms': velocidad,
                    'densidad_kg_m3': densidad,
                    'viscosidad_Pa_s': viscosidad,
                    'aspect_ratio': aspect_ratio,
                    'oswald_e': oswald_e,
                    'cd0': cd0
                },
                'coordenadas': {
                    'x': x_b.round(6).tolist(),
                    'xu': xu_b.round(6).tolist(),
                    'yu': yu_b.round(6).tolist(),
                    'xl': xl_b.round(6).tolist(),
                    'yl': yl_b.round(6).tolist()
                }
            }
            st.download_button(
                label="Descargar JSON",
                data=json.dumps(json_data_b, indent=2),
                file_name=f"NACA_{naca_b}_{cuerda_b}mm.json",
                mime="application/json"
            )

elif modo == "Visualización 3D":
    st.sidebar.subheader("Configuración del Ala")
    naca_3d = st.sidebar.text_input("Código NACA", value="2412", max_chars=5)
    envergadura = st.sidebar.slider("Envergadura (mm)", 100, 1000, 400, 50)
    cuerda_raiz = st.sidebar.slider("Cuerda raíz (mm)", 50, 200, 100, 10)
    cuerda_punta = st.sidebar.slider("Cuerda punta (mm)", 20, 150, 50, 10)
    num_secciones = st.sidebar.slider("Número de secciones", 5, 30, 15, 5)
    
    param1_3d, param2_3d, t_3d, tipo_3d = parsear_naca(naca_3d)
    
    if param1_3d is None:
        st.error(f"Código NACA '{naca_3d}' inválido. Use 4 o 5 dígitos (ej: 0015, 2412, 23012)")
        st.stop()
    
    st.subheader(f"Ala 3D con Perfil NACA {naca_3d}")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, z in enumerate(np.linspace(0, envergadura, num_secciones)):
        cuerda = cuerda_raiz + (cuerda_punta - cuerda_raiz) * (z / envergadura)
        
        x = np.linspace(0, cuerda, 100)
        xu, yu, xl, yl, yc, yt = generar_perfil_naca(x, cuerda, param1_3d, param2_3d, t_3d, tipo_3d)
        
        z_coords = np.full_like(xu, z)
        
        ax.plot(xu, z_coords, yu, 'b', linewidth=1, alpha=0.8)
        ax.plot(xl, z_coords, yl, 'r', linewidth=1, alpha=0.8)
        
        if i == 0 or i == num_secciones - 1:
            ax.plot(xu, z_coords, yu, 'b', linewidth=2)
            ax.plot(xl, z_coords, yl, 'r', linewidth=2)
    
    x_le = np.zeros(num_secciones)
    x_te_vals = []
    z_vals = np.linspace(0, envergadura, num_secciones)
    for z in z_vals:
        cuerda = cuerda_raiz + (cuerda_punta - cuerda_raiz) * (z / envergadura)
        x_te_vals.append(cuerda)
    
    ax.plot(x_le, z_vals, np.zeros(num_secciones), 'k', linewidth=2, label='Borde de ataque')
    ax.plot(x_te_vals, z_vals, np.zeros(num_secciones), 'k--', linewidth=2, label='Borde de salida')
    
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Envergadura (mm)')
    ax.set_zlabel('y (mm)')
    ax.set_title(f'Ala 3D - NACA {naca_3d}')
    
    max_range = max(envergadura, cuerda_raiz) / 2
    ax.set_box_aspect([cuerda_raiz/envergadura, 1, 0.3])
    
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("---")
    
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown("**Geometría del Ala**")
        st.write(f"- Envergadura: {envergadura} mm")
        st.write(f"- Cuerda raíz: {cuerda_raiz} mm")
        st.write(f"- Cuerda punta: {cuerda_punta} mm")
        st.write(f"- Estrechamiento: {cuerda_punta/cuerda_raiz:.2f}")
    
    with info_col2:
        area_ala = (cuerda_raiz + cuerda_punta) * envergadura / 2
        cuerda_media = (cuerda_raiz + cuerda_punta) / 2
        aspect_ratio = envergadura**2 / area_ala
        st.markdown("**Parámetros Calculados**")
        st.write(f"- Área del ala: {area_ala:.0f} mm²")
        st.write(f"- Espesor máximo: {props_a['espesor_max']:.2f} mm") 
        st.write(f"- Cuerda media: {cuerda_media:.1f} mm")
        st.write(f"- Relación de aspecto: {aspect_ratio:.2f}")

elif modo == "Importar Perfil":
    st.subheader("Importar Perfil Personalizado")
    
    st.markdown("""
    Suba un archivo con coordenadas del perfil. Formatos soportados:
    - **CSV**: Columnas `x`, `y_upper`, `y_lower` (en mm)
    - **DAT/TXT**: Formato Selig (x y, de extradós a intradós)
    """)
    
    uploaded_file = st.file_uploader("Seleccione archivo", type=['csv', 'dat', 'txt'])
    
    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name
            
            if file_name.endswith('.csv'):
                df_custom = pd.read_csv(uploaded_file)
                
                if 'x' in df_custom.columns and 'y_upper' in df_custom.columns:
                    x_custom = df_custom['x'].values
                    y_upper = df_custom['y_upper'].values
                    y_lower = df_custom['y_lower'].values if 'y_lower' in df_custom.columns else -y_upper
                else:
                    st.error("El CSV debe tener columnas 'x' y 'y_upper' (y opcionalmente 'y_lower')")
                    st.stop()
            else:
                content = uploaded_file.read().decode('utf-8')
                lines = content.strip().split('\n')
                
                coords = []
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x_val = float(parts[0])
                            y_val = float(parts[1])
                            coords.append((x_val, y_val))
                        except ValueError:
                            continue
                
                if len(coords) == 0:
                    st.error("No se pudieron leer coordenadas del archivo")
                    st.stop()
                
                coords = np.array(coords)
                
                mid_idx = len(coords) // 2
                x_upper = coords[:mid_idx+1, 0]
                y_upper = coords[:mid_idx+1, 1]
                x_lower = coords[mid_idx:, 0]
                y_lower = coords[mid_idx:, 1]
                
                x_custom = np.linspace(0, 1, 100)
                y_upper = np.interp(x_custom, x_upper[::-1], y_upper[::-1])
                y_lower = np.interp(x_custom, x_lower, y_lower)
            
            escala = st.sidebar.slider("Escala (cuerda en mm)", 50, 200, 100, 10)
            
            x_scaled = x_custom * escala
            y_upper_scaled = y_upper * escala
            y_lower_scaled = y_lower * escala
            
            st.subheader(f"Perfil Importado - Cuerda {escala} mm")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(x_scaled, y_upper_scaled, 'b', linewidth=2, label='Extradós')
            ax.plot(x_scaled, y_lower_scaled, 'r', linewidth=2, label='Intradós')
            ax.fill_between(x_scaled, y_upper_scaled, y_lower_scaled, alpha=0.1, color='gray')
            ax.set_xlabel('x (mm)', fontsize=12)
            ax.set_ylabel('y (mm)', fontsize=12)
            ax.set_title(f'Perfil Importado - {file_name}', fontsize=14)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            espesor = y_upper_scaled - y_lower_scaled
            espesor_max = np.max(espesor)
            pos_espesor_max = x_scaled[np.argmax(espesor)]
            
            st.markdown("---")
            st.subheader("Propiedades del Perfil Importado")
            st.write(f"- Espesor máximo: {espesor_max:.2f} mm ({espesor_max/escala*100:.1f}%)")
            st.write(f"- Posición espesor máx: {pos_espesor_max:.2f} mm ({pos_espesor_max/escala*100:.1f}%)")
            
            st.markdown("---")
            st.subheader("Exportar Perfil Escalado")
            
            df_export = pd.DataFrame({
                'x (mm)': x_scaled,
                'y_upper (mm)': y_upper_scaled,
                'y_lower (mm)': y_lower_scaled,
                'espesor (mm)': espesor
            })
            
            csv_export = df_export.round(6).to_csv(index=False)
            st.download_button(
                label="Descargar CSV",
                data=csv_export,
                file_name=f"perfil_importado_{escala}mm.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")

st.markdown("---")
st.markdown("""
### Sobre los Perfiles NACA

#### Perfiles de 4 Dígitos
- **1er dígito**: Curvatura máxima (% de la cuerda)
- **2do dígito**: Posición de la curvatura máxima (décimas de cuerda)
- **3er y 4to dígitos**: Espesor máximo (% de la cuerda)

**Ejemplos:**
- **NACA 0015**: Simétrico, 15% de espesor
- **NACA 2412**: 2% curvatura a 40% de la cuerda, 12% espesor
- **NACA 4415**: 4% curvatura a 40% de la cuerda, 15% espesor

#### Perfiles de 5 Dígitos
- **1er dígito**: Coeficiente de sustentación de diseño (×0.15)
- **2do dígito**: Posición de curvatura máxima (×0.05 de la cuerda)
- **3er dígito**: 0 = curvatura normal, 1 = curvatura reflex
- **4to y 5to dígitos**: Espesor máximo (% de la cuerda)

**Ejemplos:**
- **NACA 23012**: Cl=0.3, curvatura a 15%, 12% espesor
- **NACA 24112**: Cl=0.3, curvatura a 20%, reflex, 12% espesor
""")
