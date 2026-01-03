import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# --- הגדרות עמוד ועיצוב ---
st.set_page_config(
    layout="wide",
    page_title="Animal Shelter Pro Dashboard",
    page_icon="🐾",
    initial_sidebar_state="expanded"
)

# עיצוב מותאם אישית
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #FF4B4B; text-align: center; font-weight: bold; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.5rem; color: #333; margin-top: 1.5rem; border-bottom: 2px solid #FF4B4B; margin-bottom: 1rem;}
    div.stButton > button {background-color: #FF4B4B; color: white;}
    .stRadio > label {font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- טעינת נתונים ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('clean_data (2).csv')
    except FileNotFoundError:
        st.error("⚠️ הקובץ 'clean_data (2).csv' לא נמצא.")
        st.stop()
    
    # המרת תאריכים
    cols_to_date = ['intake_datetime', 'outcome_datetime', 'date_of_birth']
    for col in cols_to_date:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # יצירת עמודות עזר
    if 'intake_datetime' in df.columns:
        df['Intake Month'] = df['intake_datetime'].dt.month_name()
        df['Intake Day'] = df['intake_datetime'].dt.day_name()
        df['Intake Hour'] = df['intake_datetime'].dt.hour
        df['intake_year'] = df['intake_datetime'].dt.year
    
    # ערכים חסרים וניקוי
    df['outcome_subtype'] = df['outcome_subtype'].fillna('Unknown')
    if 'sex' not in df.columns and 'sex_upon_outcome' in df.columns:
        df['sex'] = df['sex_upon_outcome'] # גיבוי אם העמודה חסרה
        
    return df

df = load_data()

# --- פונקציות עזר ---

def calculate_rate(df, group_col, target_col='outcome_type', target_val='Adopted'):
    """חישוב אחוזים גנרי"""
    if target_col in df.columns:
        df['is_target'] = df[target_col] == target_val
        rate = df.groupby(group_col, observed=False)['is_target'].mean().reset_index()
        rate.rename(columns={'is_target': 'Rate'}, inplace=True)
        return rate
    return pd.DataFrame()

def monthly_counts(data, animal_type):
    d = data[data["animal_type"] == animal_type].copy()
    if d.empty: return pd.Series(0, index=range(1, 13))
    counts = d.groupby(d["intake_datetime"].dt.month).size().reindex(range(1, 13), fill_value=0)
    return counts

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def plot_polar_month_spectrum(ax, month_counts, title, global_max, cmap_name="YlOrRd"):
    values = month_counts.values.astype(float)
    months = np.arange(12)
    theta = (2 * np.pi) * (months / 12.0)
    width = (2 * np.pi) / 12.0 * 0.92
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=0, vmax=global_max)
    colors = cmap(norm(values))
    ax.bar(theta, values, width=width, bottom=0, color=colors, edgecolor="white", linewidth=1.0)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    if global_max > 0: ax.set_ylim(0, global_max * 1.05)
    ax.set_xticks(theta)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_yticks([])
    ax.grid(alpha=0.3)
    # Winter highlight
    winter_indices = [11, 0, 1]
    for idx in winter_indices:
        t0 = theta[idx] - width/2
        ax.bar(x=t0 + width/2, height=global_max*1.05, width=width, bottom=0, color=(0,0,0,0.03), edgecolor=None)

# --- ניווט ---
st.sidebar.title("🐾 Dashboard Menu")
page = st.sidebar.radio("Go to:", 
    ["1. Overview & Trends", 
     "2. Seasonality & Operations", 
     "3. Factors & Analysis", 
     "4. Population Stats"])

st.markdown(f"<div class='main-header'>{page.split('. ')[1]}</div>", unsafe_allow_html=True)

# ==========================================
# PAGE 1: OVERVIEW & TRENDS
# ==========================================
if "1." in page:
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")
    
    # 1. פילטרים בסיסיים
    min_y, max_y = int(df['intake_year'].min()), int(df['intake_year'].max())
    selected_years = st.sidebar.slider("Year Range:", min_y, max_y, (min_y, max_y))
    
    animal_types = df['animal_type'].unique()
    selected_animals = st.sidebar.multiselect("Animal Type:", animal_types, default=animal_types)

    # 2. [NEW] פילטרים מתקדמים בתוך Expander
    with st.sidebar.expander("➕ Advanced Filters"):
        intake_opts = df['intake_type'].unique()
        selected_intake = st.multiselect("Intake Type:", intake_opts, default=intake_opts)
        
        sex_opts = df['sex'].unique() if 'sex' in df.columns else []
        selected_sex = st.multiselect("Sex:", sex_opts, default=sex_opts)

    # סינון
    mask = (df['intake_year'] >= selected_years[0]) & \
           (df['intake_year'] <= selected_years[1]) & \
           (df['animal_type'].isin(selected_animals)) & \
           (df['intake_type'].isin(selected_intake))
    
    if selected_sex:
        mask = mask & (df['sex'].isin(selected_sex))
        
    filtered_df = df[mask]

    # מדדים
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Intakes", f"{len(filtered_df):,}")
    c2.metric("Adoption Rate", f"{(filtered_df['outcome_type'] == 'Adopted').mean():.1%}")
    c3.metric("Transfer Rate", f"{(filtered_df['outcome_type'] == 'Transfer').mean():.1%}")
    avg_days = filtered_df['time_in_shelter_days'].mean()
    c4.metric("Avg Stay (Days)", f"{avg_days:.1f}")

    st.markdown("---")

    # [NEW] אינטראקטיביות בזמן
    st.subheader("📈 Intake Trends")
    c_ctrl, c_chart = st.columns([1, 4])
    with c_ctrl:
        time_grain = st.radio("Time Granularity:", ["Monthly", "Quarterly", "Yearly"])
        grain_map = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
    
    with c_chart:
        trend_data = filtered_df.set_index('intake_datetime').resample(grain_map[time_grain]).size().reset_index(name='Count')
        fig_trend = px.line(trend_data, x='intake_datetime', y='Count', markers=True, 
                            color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig_trend, use_container_width=True)

    # Sunburst
    st.subheader("☀️ Outcome Breakdown")
    sb_df = filtered_df.groupby(['outcome_type', 'outcome_subtype'], observed=False).size().reset_index(name='count')
    sb_df = sb_df[sb_df['count'] > 5] # סינון רעשים
    fig_sun = px.sunburst(sb_df, path=['outcome_type', 'outcome_subtype'], values='count', 
                          color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_sun.update_layout(height=500)
    st.plotly_chart(fig_sun, use_container_width=True)

# ==========================================
# PAGE 2: SEASONALITY
# ==========================================
elif "2." in page:
    st.sidebar.markdown("---")
    st.sidebar.header("❄️ Settings")
    
    polar_year = st.sidebar.selectbox("Select Year for Polar Plot:", sorted(df['intake_year'].unique(), reverse=True))
    
    # [NEW] פילטר חיה לכל העמוד
    focus_animal = st.sidebar.radio("Focus Animal for Heatmaps:", ["All", "Dog", "Cat"])

    st.subheader(f"❄️ Seasonal Patterns ({polar_year})")
    
    df_year = df[df["intake_year"] == polar_year].copy()
    if not df_year.empty:
        # Polar Plots תמיד מראים השוואה
        cats_m = monthly_counts(df_year, "Cat")
        dogs_m = monthly_counts(df_year, "Dog")
        max_val = max(cats_m.max(), dogs_m.max())
        
        fig_polar, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "polar"})
        plot_polar_month_spectrum(axes[0], cats_m, "Cats", max_val)
        plot_polar_month_spectrum(axes[1], dogs_m, "Dogs", max_val)
        st.pyplot(fig_polar)
    
    st.markdown("---")
    
    # Heatmaps מושפעים מהפוקוס
    ops_df = df.copy()
    if focus_animal != "All":
        ops_df = ops_df[ops_df['animal_type'] == focus_animal]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📅 Weekly Heatmap")
        pivot = ops_df.groupby(['Intake Month', 'Intake Day'], observed=False).size().unstack().fillna(0)
        days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        pivot = pivot.reindex(index=[m for m in months if m in pivot.index], columns=[d for d in days if d in pivot.columns])
        fig_cal = px.imshow(pivot, labels=dict(x="Day", y="Month", color="Intakes"), color_continuous_scale="Reds")
        st.plotly_chart(fig_cal, use_container_width=True)
    
    with c2:
        st.subheader("🕒 Peak Hours")
        hourly = ops_df.groupby('Intake Hour', observed=False).size().reset_index(name='Count')
        fig_area = px.area(hourly, x='Intake Hour', y='Count', markers=True, color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig_area, use_container_width=True)

# ==========================================
# PAGE 3: FACTORS
# ==========================================
elif "3." in page:
    st.sidebar.markdown("---")
    st.sidebar.header("🧬 Filters")
    
    # פילטרים קיימים
    breed_filter = st.sidebar.radio("Breed Purity:", ["All", "Mixed", "Purebred"])
    
    # [NEW] פילטר גיל
    with st.sidebar.expander("➕ Filter by Age Group"):
        age_opts = df['age_group'].unique()
        selected_ages = st.multiselect("Age Groups:", age_opts, default=age_opts)

    factor_df = df.copy()
    if breed_filter != "All" and 'mixed_purebred' in factor_df.columns:
        factor_df = factor_df[factor_df['mixed_purebred'] == breed_filter]
    if selected_ages:
        factor_df = factor_df[factor_df['age_group'].isin(selected_ages)]

    # Heatmap קבוע (הכי חשוב)
    st.subheader("🔥 Adoption Probability: Age vs Animal")
    heatmap_data = factor_df.groupby(['animal_type', 'age_group'], observed=False)['outcome_type'].value_counts(normalize=True).unstack().fillna(0)
    if 'Adopted' in heatmap_data.columns:
        hm_piv = heatmap_data['Adopted'].reset_index().pivot(index='animal_type', columns='age_group', values='Adopted')
        age_order = ['Puppy_Kitten', 'Juvenile', 'Adult', 'Senior']
        cols = [c for c in age_order if c in hm_piv.columns]
        if cols: hm_piv = hm_piv[cols]
        fig_h, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(hm_piv, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
        st.pyplot(fig_h)

    st.markdown("---")

    # [NEW] Dynamic Factor Analysis
    st.subheader("🕵️ Explore Factors")
    st.caption("Select a factor to analyze its impact on adoption rates.")
    
    factor_choice = st.selectbox("Analyze Adoption By:", 
                                 ["Sex & Neuter Status", "Health Status", "Color Group", "Intake Type"])
    
    if factor_choice == "Sex & Neuter Status":
        if 'sex' in factor_df.columns:
            factor_df['Status'] = factor_df['neutered'].map({True: 'Neutered', False: 'Intact'})
            rate = calculate_rate(factor_df, ['sex', 'Status'])
            fig = px.bar(rate, x='sex', y='Rate', color='Status', barmode='group', 
                         color_discrete_map={'Neutered': '#2ca02c', 'Intact': '#d62728'}, text_auto='.1%')
            st.plotly_chart(fig, use_container_width=True)
            
    elif factor_choice == "Health Status":
        rate = calculate_rate(factor_df, ['health_status'])
        fig = px.bar(rate, x='health_status', y='Rate', color='health_status', text_auto='.1%')
        st.plotly_chart(fig, use_container_width=True)

    elif factor_choice == "Color Group":
        rate = calculate_rate(factor_df, ['color_group'])
        fig = px.bar(rate.sort_values('Rate'), x='Rate', y='color_group', orientation='h', 
                     color='Rate', color_continuous_scale='Bluyl')
        st.plotly_chart(fig, use_container_width=True)
        
    elif factor_choice == "Intake Type":
        rate = calculate_rate(factor_df, ['intake_type'])
        fig = px.bar(rate, x='intake_type', y='Rate', color='intake_type', text_auto='.1%')
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 4: POPULATION
# ==========================================
elif "4." in page:
    st.sidebar.markdown("---")
    st.sidebar.header("📍 Filters")
    
    # [NEW] Multi-filters
    selected_intake = st.sidebar.multiselect("Intake Source:", df['intake_type'].unique(), default=['Stray', 'Owner Surrender'])
    
    with st.sidebar.expander("➕ More Filters"):
        selected_health = st.multiselect("Health Status:", df['health_status'].unique(), default=df['health_status'].unique())
        selected_sex = st.multiselect("Sex:", df['sex'].unique(), default=df['sex'].unique())

    pop_df = df[df['intake_type'].isin(selected_intake) & 
                df['health_status'].isin(selected_health) & 
                df['sex'].isin(selected_sex)]

    st.subheader("🎻 Age Distribution (Violin Plot)")
    fig_violin = plt.figure(figsize=(10, 5))
    sns.violinplot(data=pop_df, x="outcome_type", y="age_norm", hue="animal_type", 
                   split=True, inner="quart", palette="muted")
    st.pyplot(fig_violin)

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📍 Top Found Locations")
        if 'found_location' in pop_df.columns:
            top_locs = pop_df['found_location'].value_counts().head(10).reset_index()
            top_locs.columns = ['Location', 'Count']
            fig_loc = px.bar(top_locs, x='Count', y='Location', orientation='h', title="Top 10 Locations")
            fig_loc.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_loc, use_container_width=True)
            
    with c2:
        st.subheader("⏳ Shelter Time Distribution")
        # [NEW] שליטה על ה-bins
        nbins = st.slider("Histogram Bins:", 10, 100, 40)
        fig_hist = px.histogram(pop_df, x="time_in_shelter_days", nbins=nbins, color="animal_type", 
                                marginal="box", range_x=[0, 100])
        st.plotly_chart(fig_hist, use_container_width=True)