import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import datetime
import calendar

# --- PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="Paws & Data: Shelter Analysis",
    page_icon="🐶",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS & STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }
    
    /* רקע כללי רך */
    .stApp {
        background-color: #FDFBF7;
    }
    
    /* כותרת ראשית מעוצבת */
    .main-header {
        font-size: 3rem; 
        color: #FF6B6B; 
        text-align: center; 
        font-weight: 800; 
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px #e0e0e0;
        margin-top: 1rem; /* הוספת מרווח עליון במקום התמונה */
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* עיצוב כרטיסיות (Cards) */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* כותרות גרפים */
    .chart-header {
        font-size: 1.5rem; 
        color: #4ECDC4; 
        margin-top: 1rem; 
        border-bottom: 3px solid #FF6B6B; 
        padding-bottom: 5px;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    /* כפתורים */
    div.stButton > button {
        background-color: #FF6B6B; 
        color: white; 
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #EE5253;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('clean_data (2).csv')
    except FileNotFoundError:
        st.error("⚠️ File 'clean_data (2).csv' not found.")
        st.stop()
    
    cols_to_date = ['intake_datetime', 'outcome_datetime', 'date_of_birth']
    for col in cols_to_date:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    if 'intake_datetime' in df.columns:
        df['intake_year'] = df['intake_datetime'].dt.year
        df['intake_month'] = df['intake_datetime'].dt.month
        df['date'] = df['intake_datetime'].dt.date
    
    if 'color_group' in df.columns:
        df['color'] = df['color_group']
    else:
        df['color'] = 'Unknown'

    df['outcome_type'] = df['outcome_type'].fillna('Unknown')
    return df

df = load_data()

# --- HELPER FUNCTIONS & COLORS ---
color_map = {
    'Black': '#2d3436', 'Black/White': '#2d3436', 'White': '#dfe6e9',
    'Brown': '#8B4513', 'Brown Tabby': '#d35400', 'Tan': '#e1b12c',
    'Brown/White': '#A0522D', 'White/Black': '#b2bec3', 'Tan/White': '#ffeaa7',
    'Tricolor': '#e67e22', 'Blue': '#74b9ff', 'Blue/White': '#a29bfe',
    'Orange Tabby': '#fab1a0', 'Brown Tabby/White': '#e17055', 'Black/Tan': '#2d3436',
    'Calico': '#fdcb6e', 'Red': '#ff7675', 'Torbie': '#d63031',
    'Cream Tabby': '#fab1a0', 'Gray': '#636e72', 'Chocolate': '#6D214F',
    'Fawn': '#ffeaa7', 'Buff': '#fdcb6e', 'Yellow': '#ffeaa7',
    'Cream': '#fff7d1', 'Lynx Point': '#dfe6e9', 'Seal Point': '#2d3436',
    'Flame Point': '#fab1a0', 'Blue Point': '#74b9ff', 'Light': '#fdcb6e'
}

def get_hex(name):
    if not isinstance(name, str): return '#95a5a6'
    if name in color_map: return color_map[name]
    lower = name.lower()
    if 'white' in lower: return '#dfe6e9'
    if 'black' in lower: return '#2d3436'
    if 'brown' in lower: return '#8B4513'
    if 'blue' in lower: return '#74b9ff'
    if 'orange' in lower: return '#ff7675'
    if 'gray' in lower: return '#636e72'
    return '#b2bec3'

def get_luminance(hex_color):
    try:
        rgb = mcolors.hex2color(hex_color)
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    except: return 0.5

display_names = {
    'Black/White': 'Tuxedo', 'White/Black': 'White & Patches',
    'Tan/White': 'Golden Cream', 'Brown Tabby': 'Classic Tiger',
    'Orange Tabby': 'Ginger Striped', 'Light': 'Cream/White'
}

def clean_label(name):
    return display_names.get(name, name.replace('/', ' & ').replace(' Mix', ''))

# --- PLOTTING LOGIC ---

def calculate_fixed_dots(df_subset, animals_per_dot):
    total_animals = len(df_subset)
    if total_animals == 0: return [], 0
    target_total_dots = int(total_animals / animals_per_dot)
    counts = df_subset['color'].value_counts()
    df_calc = counts.reset_index()
    df_calc.columns = ['color', 'count']
    df_calc['exact_dots'] = df_calc['count'] / animals_per_dot
    df_calc['floor_dots'] = df_calc['exact_dots'].apply(np.floor).astype(int)
    missing_dots = target_total_dots - df_calc['floor_dots'].sum()
    if missing_dots > 0: df_calc.iloc[:missing_dots, df_calc.columns.get_loc('floor_dots')] += 1
    df_calc['hex'] = df_calc['color'].apply(get_hex)
    df_calc['luminance'] = df_calc['hex'].apply(get_luminance)
    df_calc = df_calc.sort_values(by='luminance', ascending=True)
    final_list = []
    for _, row in df_calc.iterrows():
        final_list.extend([row['hex']] * int(row['floor_dots']))
    return final_list, total_animals

def plot_fixed_grid(ax, color_list, title, total_count, grid_width=10):
    if not color_list: 
        ax.text(0.5, 0.5, "No Data", ha='center'); ax.axis('off'); return
    x_list = [i % grid_width for i in range(len(color_list))]
    y_list = [i // grid_width for i in range(len(color_list))]
    ax.scatter(x_list, y_list, c=color_list, s=180, edgecolors='#555', linewidth=0.5, alpha=0.9)
    ax.set_title(f'{title}\n({total_count:,})', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.axis('off')
    if y_list: ax.set_ylim(-1, max(y_list) + 1.5)
    ax.set_xlim(-1, grid_width)

def plot_spectrum_subplot(ax, df_subset, animal_name, show_all=False, selected_colors=[]):
    if df_subset.empty: ax.axis('off'); return
    if show_all: color_counts = df_subset['color'].value_counts()
    elif selected_colors: 
        df_subset = df_subset[df_subset['color'].isin(selected_colors)]
        color_counts = df_subset['color'].value_counts()
    else: color_counts = df_subset['color'].value_counts().head(20)
    
    if color_counts.empty: ax.axis('off'); return
    df_colors = color_counts.reset_index()
    df_colors.columns = ['original_name', 'count']
    df_colors['hex'] = df_colors['original_name'].apply(get_hex)
    df_colors['luminance'] = df_colors['hex'].apply(get_luminance)
    df_colors['display_name'] = df_colors['original_name'].apply(clean_label)
    df_colors = df_colors.sort_values(by='luminance', ascending=True)
    
    bars = ax.barh(df_colors['display_name'], df_colors['count'], color=df_colors['hex'], edgecolor='gray', height=0.7)
    ax.set_title(f'{animal_name}', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.2)
    for bar, count in zip(bars, df_colors['count']):
        ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height()/2, f'{count:,}', va='center', fontsize=9, fontweight='bold', color='#555')

def get_cal_matrix(data_subset, year):
    df_year = data_subset[data_subset['intake_datetime'].dt.year == year]
    daily_counts = df_year.groupby('date').size()
    start, end = datetime.date(year, 1, 1), datetime.date(year, 12, 31)
    daily_counts = daily_counts.reindex(pd.date_range(start, end).date, fill_value=0)
    matrix = np.zeros((7, 54))
    for date, count in daily_counts.items():
        matrix[int(date.strftime('%w')), int(date.strftime('%U'))] = count
    return matrix, daily_counts.max()

def monthly_counts(data, animal_type):
    d = data[data["animal_type"] == animal_type].copy()
    if d.empty: return pd.Series(0, index=range(1, 13))
    return d.groupby(d["intake_datetime"].dt.month).size().reindex(range(1, 13), fill_value=0)

def plot_polar_month_spectrum(ax, month_counts, title, global_max):
    values = month_counts.values.astype(float)
    theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
    width = 2*np.pi / 12 * 0.9
    cmap = plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=0, vmax=global_max)
    ax.bar(theta, values, width=width, bottom=0, color=cmap(norm(values)), edgecolor="white", alpha=0.9)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10, color='#2c3e50')
    if global_max > 0: ax.set_ylim(0, global_max * 1.05)
    ax.set_xticks(theta)
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=9)
    ax.set_yticks([])

# --- SIDEBAR WITH IMAGE ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1548199973-03cce0bbc87b?q=80&w=400&auto=format&fit=crop", use_container_width=True)
    st.title("🐾 Controls")
    min_year, max_year = int(df['intake_year'].min()), int(df['intake_year'].max())
    selected_years = st.slider("📆 Time Period:", min_year, max_year, (min_year, max_year))
    all_animals = df['animal_type'].unique()
    selected_animals = st.multiselect("🐶 Filter Animals:", all_animals, default=all_animals)
    st.markdown("---")
    page = st.radio("📚 Navigation:", [
        "1. Overview (Dot Matrix)",
        "2. Color Spectrum",
        "3. Seasonality Patterns",
        "4. Deep Analysis"
    ])
    st.markdown("---")
    st.caption("Data Visualization Project")

# --- FILTER DATA ---
global_mask = (df['intake_year'] >= selected_years[0]) & \
              (df['intake_year'] <= selected_years[1]) & \
              (df['animal_type'].isin(selected_animals))
filtered_df = df[global_mask]

if filtered_df.empty:
    st.warning("⚠️ No data available for these filters. Please adjust the sidebar.")
    st.stop()

# --- MAIN PAGE HEADER & METRICS ---
# התמונה שהייתה כאן נמחקה

st.markdown(f"<div class='main-header'>Animal Shelter Analysis</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>Exploring intake trends, colors, and outcomes from {selected_years[0]} to {selected_years[1]}</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
total_count = len(filtered_df)
dogs_count = len(filtered_df[filtered_df['animal_type']=='Dog'])
cats_count = len(filtered_df[filtered_df['animal_type']=='Cat'])
adoption_rate = len(filtered_df[filtered_df['outcome_type']=='Adopted']) / total_count * 100 if total_count > 0 else 0

col1.metric("🐾 Total Animals", f"{total_count:,}")
col2.metric("🐶 Dogs", f"{dogs_count:,}")
col3.metric("🐱 Cats", f"{cats_count:,}")
col4.metric("🏠 Adoption Rate", f"{adoption_rate:.1f}%")

st.divider()

# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if "1." in page:
    st.markdown("<div class='chart-header'>✨ The Scale of Intakes</div>", unsafe_allow_html=True)
    st.info("Each dot represents a **group** of animals. Adjust the slider to see the volume difference.")
    scale = st.slider("Animals per Dot:", 100, 2000, 1000, step=100)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#FDFBF7')
    if "Dog" in selected_animals:
        dogs_data = filtered_df[filtered_df['animal_type'] == 'Dog']
        d_colors, d_total = calculate_fixed_dots(dogs_data, scale)
        plot_fixed_grid(axes[0], d_colors, "Dogs", d_total)
    else: axes[0].axis('off')
    if "Cat" in selected_animals:
        cats_data = filtered_df[filtered_df['animal_type'] == 'Cat']
        c_colors, c_total = calculate_fixed_dots(cats_data, scale)
        plot_fixed_grid(axes[1], c_colors, "Cats", c_total)
    else: axes[1].axis('off')
    st.pyplot(fig)

# ==========================================
# PAGE 2: DEMOGRAPHICS
# ==========================================
elif "2." in page:
    st.markdown("<div class='chart-header'>🎨 Color Spectrum Analysis</div>", unsafe_allow_html=True)
    col_sel, col_check = st.columns([3, 1])
    with col_sel:
        all_colors = sorted(filtered_df['color'].unique().astype(str))
        chosen_colors = st.multiselect("Filter Specific Colors:", all_colors)
    with col_check:
        st.write("")
        st.write("")
        show_all = st.checkbox("Show ALL Colors", value=False)
    
    temp_dog = filtered_df[filtered_df['animal_type'] == 'Dog']
    temp_cat = filtered_df[filtered_df['animal_type'] == 'Cat']
    max_bars = 20
    if show_all: max_bars = max(len(temp_dog['color'].unique()), len(temp_cat['color'].unique()))
    elif chosen_colors: max_bars = len(chosen_colors)
    fig_height = max(6, max_bars * 0.4)
    fig, axes = plt.subplots(1, 2, figsize=(14, fig_height))
    fig.patch.set_facecolor('#FDFBF7')
    if "Dog" in selected_animals: plot_spectrum_subplot(axes[0], temp_dog, 'Dogs', show_all, chosen_colors)
    else: axes[0].axis('off')
    if "Cat" in selected_animals: plot_spectrum_subplot(axes[1], temp_cat, 'Cats', show_all, chosen_colors)
    else: axes[1].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# PAGE 3: SEASONALITY
# ==========================================
elif "3." in page:
    st.markdown("<div class='chart-header'>📅 Seasonal Intake Patterns</div>", unsafe_allow_html=True)
    valid_years = sorted(filtered_df['intake_year'].unique())
    target_year = st.selectbox("Select Focus Year:", valid_years, index=0)
    st.subheader(f"Daily Heatmap ({target_year})")
    matrix_cats, max_cat = get_cal_matrix(filtered_df[filtered_df['animal_type'] == 'Cat'], target_year)
    matrix_dogs, max_dog = get_cal_matrix(filtered_df[filtered_df['animal_type'] == 'Dog'], target_year)
    global_max = max(max_cat, max_dog)
    fig_cal, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig_cal.patch.set_facecolor('#FDFBF7')
    heatmap_cmap = 'OrRd'
    if "Cat" in selected_animals:
        sns.heatmap(matrix_cats, ax=axes[0], cmap=heatmap_cmap, linewidths=0.5, linecolor='white', square=True,
                    vmin=0, vmax=global_max, cbar=False)
        axes[0].set_title('🐱 Cats Intake', fontweight='bold', fontsize=12, loc='left')
        axes[0].set_xticks([]); axes[0].set_yticklabels(['S','M','T','W','T','F','S'], rotation=0)
    else: axes[0].axis('off')
    if "Dog" in selected_animals:
        sns.heatmap(matrix_dogs, ax=axes[1], cmap=heatmap_cmap, linewidths=0.5, linecolor='white', square=True,
                    vmin=0, vmax=global_max, cbar=False)
        axes[1].set_title('🐶 Dogs Intake', fontweight='bold', fontsize=12, loc='left')
        axes[1].set_yticklabels(['S','M','T','W','T','F','S'], rotation=0)
        month_starts = [datetime.date(target_year, m, 1).strftime('%U') for m in range(1, 13)]
        axes[1].set_xticks([int(w) + 1.5 for w in month_starts])
        axes[1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], rotation=0)
    else: axes[1].axis('off')
    st.pyplot(fig_cal)
    st.markdown("---")
    st.subheader(f"❄️ vs ☀️ Monthly Cycle")
    cats_m = monthly_counts(filtered_df[filtered_df['intake_year'] == target_year], "Cat")
    dogs_m = monthly_counts(filtered_df[filtered_df['intake_year'] == target_year], "Dog")
    g_max_pol = int(max(cats_m.max(), dogs_m.max()))
    fig_pol, ax_pol = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "polar"})
    fig_pol.patch.set_facecolor('#FDFBF7')
    if "Cat" in selected_animals: plot_polar_month_spectrum(ax_pol[0], cats_m, f"Cats Cycle", g_max_pol)
    else: ax_pol[0].axis('off')
    if "Dog" in selected_animals: plot_polar_month_spectrum(ax_pol[1], dogs_m, f"Dogs Cycle", g_max_pol)
    else: ax_pol[1].axis('off')
    st.pyplot(fig_pol)

# ==========================================
# PAGE 4: ANALYSIS
# ==========================================
elif "4." in page:
    st.markdown("<div class='chart-header'>📊 Analysis: Trends & Outcomes</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    
    # 4a. Puppy Surrenders (The "Season of Regret")
    with col_a:
        st.subheader("🥀 The 'Season of Regret'")
        st.caption("Puppy surrenders peak in Jan/Feb and drop in Summer")
        
        if "Dog" in selected_animals:
            df_surrender = filtered_df[(filtered_df['animal_type'] == 'Dog') & (filtered_df['intake_type'] == 'Owner Surrender')].copy()
            puppy_surrenders = df_surrender[df_surrender['age_group'].isin(['Puppy', 'Puppy_Kitten'])]
            
            if not puppy_surrenders.empty:
                counts = puppy_surrenders.groupby(puppy_surrenders['intake_datetime'].dt.month).size().reindex(range(1, 13), fill_value=0)
                months = np.arange(1, 13)
                values = counts.values
                max_val = values.max(); min_val = values.min() if values.min() > 0 else 1
                max_month_idx = values.argmax() + 1; min_month_idx = values.argmin() + 1
                avg_val = values.mean(); ratio = max_val / min_val
                
                # --- שינוי: הגדלת הגרף כאן ---
                fig_area, ax = plt.subplots(figsize=(14, 8)) # הוגדל מ-(10, 6)
                fig_area.patch.set_facecolor('#FDFBF7')
                
                ax.plot(months, values, color='#c0392b', linewidth=2.5, zorder=3)
                ax.fill_between(months, values, color='#e74c3c', alpha=0.2)
                winter_mask = (months <= 2) | (months >= 11)
                ax.fill_between(months, values, where=winter_mask, color='#c0392b', alpha=0.3, interpolate=True)
                ax.axhline(y=avg_val, color='gray', linestyle='--', alpha=0.6, linewidth=1)
                ax.text(12.2, avg_val, f'Yearly Avg ({avg_val:.1f})', verticalalignment='center', color='gray', fontsize=8)

                ax.annotate(f'The "Regret" Peak\n(Jan: {max_val} Puppies)', 
                            xy=(max_month_idx, max_val), xytext=(max_month_idx + 2, max_val + 2),
                            arrowprops=dict(facecolor='#c0392b', shrink=0.05, width=1.5, headwidth=8),
                            fontsize=9, fontweight='bold', color='#c0392b')

                ax.annotate(f'The Summer Low\n({calendar.month_abbr[min_month_idx]}: Only {min_val})', 
                            xy=(min_month_idx, min_val), xytext=(min_month_idx - 1, min_val + 10),
                            arrowprops=dict(facecolor='#2980b9', shrink=0.05, width=1.5, headwidth=8),
                            fontsize=9, fontweight='bold', color='#2980b9', ha='center')

                text_content = f"Puppy surrenders are\n{ratio:.1f}x higher in Winter\nthan in Summer!"
                ax.text(6.5, max_val * 0.85, text_content, 
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9),
                        ha='center', fontsize=10, fontweight='bold', color='#2c3e50')

                ax.set_xticks(months)
                ax.set_xticklabels([calendar.month_abbr[i] for i in months], fontsize=9)
                ax.set_ylim(0, max_val * 1.2)
                ax.set_ylabel("Number of Surrendered Puppies")
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                st.pyplot(fig_area)
            else: st.info("Not enough puppy surrender data for this selection.")
        else: st.warning("Select 'Dogs' in the sidebar to view this chart.")

    # 4b. Adoption Rates
    with col_b:
        st.subheader("🏠 Adoption Success")
        st.caption("Adoption rates by age group")
        age_compares = filtered_df.groupby(['animal_type', 'age_group'], observed=False)['outcome_type'].apply(lambda x: (x == 'Adopted').mean()).reset_index(name='Adopted')
        fig_rates, ax_rates = plt.subplots(1, 2, figsize=(8, 6))
        fig_rates.patch.set_facecolor('#FDFBF7')
        for i, animal in enumerate(['Cat', 'Dog']):
            if animal in selected_animals:
                subset = age_compares[age_compares['animal_type'] == animal]
                if not subset.empty:
                    x, a, n = subset["age_group"], subset["Adopted"]*100, (1-subset["Adopted"])*100
                    ax_rates[i].bar(x, a, color="#4ECDC4", label="Adopted", width=0.6, edgecolor='white', linewidth=1.5)
                    ax_rates[i].bar(x, n, bottom=a, color="#fab1a0", label="Other", width=0.6, edgecolor='white', linewidth=1.5)
                    ax_rates[i].set_title(animal, fontweight='bold')
                    ax_rates[i].set_ylim(0,100)
                    ax_rates[i].tick_params(axis='x', rotation=45)
                    ax_rates[i].spines['top'].set_visible(False); ax_rates[i].spines['right'].set_visible(False)
                    ax_rates[i].spines['left'].set_visible(False); ax_rates[i].get_yaxis().set_visible(False)
                    for idx, val in enumerate(a):
                        ax_rates[i].text(idx, val/2, f"{val:.0f}%", ha='center', color='white', fontweight='bold', fontsize=10)
            else: ax_rates[i].axis('off')
        st.pyplot(fig_rates)