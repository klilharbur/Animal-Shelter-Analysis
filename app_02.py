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
    
    .stApp {
        background-color: #FDFBF7;
    }
    
    .main-header {
        font-size: 3rem; 
        color: #FF6B6B; 
        text-align: center; 
        font-weight: 800; 
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px #e0e0e0;
        margin-top: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chart-header {
        font-size: 1.5rem; 
        color: #4ECDC4; 
        margin-top: 1rem; 
        border-bottom: 3px solid #FF6B6B; 
        padding-bottom: 5px;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
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
    
    .chart-explanation {
        background-color: #eafbf9;
        border-left: 5px solid #4ECDC4;
        padding: 15px;
        border-radius: 5px;
        color: #2c3e50;
        font-size: 0.95rem;
        margin-bottom: 15px;
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

# --- HELPER FUNCTIONS & COLORS (UPDATED) ---

# 1. מילון שמות "יפים"
display_names = {
    # כלבים וחתולים - שילובים נפוצים
    'Black/White': 'Tuxedo Style',
    'White/Black': 'White & Patches',
    'Tan/White': 'Golden Cream',
    'White/Tan': 'Cloudy Beige',
    'Brown/White': 'Chestnut Bicolor',
    'White/Brown': 'White & Earth',
    'Blue/White': 'Silver & White',
    'White/Blue': 'Frosty Grey',
    'Black/Tan': 'Rottweiler Pattern',
    'Tan/Black': 'Shepherd Mix',
    'Brown/Tan': 'Choco-Caramel',

    # חתולים ספציפי
    'Brown Tabby': 'Classic Tiger',
    'Orange Tabby': 'Ginger Striped',
    'Blue Tabby': 'Grey Tiger',
    'Cream Tabby': 'Soft Stripes',
    'Torbie': 'Tortoise Striped',
    'Calico': 'Tricolor Calico',
    'Tortie': 'Tortoiseshell',
    'Lynx Point': 'Siamese Mix',
    'Seal Point': 'Dark Mask',
    'Flame Point': 'Orange Mask',

    # צבעים בודדים
    'Buff': 'Champagne',
    'Sable': 'Dark Sable',
    'Fawn': 'Light Deer',
    'Tricolor': 'Three Color Mix',
    'Apricot': 'Soft Peach'
}

# 2. מילון צבעים (HEX) מורחב
color_map = {
    'Black': '#000000', 'Black/White': '#2C2C2C', 'White': '#F5F5F5',
    'Brown': '#8B4513', 'Brown Tabby': '#A0522D', 'Tan': '#D2B48C',
    'Brown/White': '#A0522D', 'White/Black': '#DCDCDC', 'Tan/White': '#FFE4B5',
    'Tricolor': '#CD853F', 'Blue': '#778899', 'Blue/White': '#B0C4DE',
    'Orange Tabby': '#FFA500', 'Brown Tabby/White': '#DEB887', 'Black/Tan': '#4B3621',
    'Black/Brown': '#3E2723', 'Tortie': '#8B5A2B', 'Calico': '#F4A460',
    'Red': '#B22222', 'Red/White': '#CD5C5C', 'Torbie': '#A0522D',
    'Cream Tabby': '#FDF5E6', 'Blue Tabby': '#708090', 'Gray': '#808080',
    'Chocolate': '#D2691E', 'Fawn': '#E6E6FA', 'Buff': '#F0E68C',
    'Yellow': '#FFD700', 'Sable': '#5C4033', 'Apricot': '#FFDAB9',
    'Cream': '#FFFDD0', 'Lynx Point': '#E0FFFF', 'Seal Point': '#3D2B1F',
    'Flame Point': '#FFE4E1', 'Blue Point': '#B0C4DE', 'Lilac Point': '#E6E6FA',
    'Tortie Point': '#8B5A2B', 'Chocolate Point': '#D2691E',
    'Light': '#fdcb6e' # Backup
}

def clean_label(name):
    if name in display_names:
        return display_names[name]
    return name.replace('/', ' & ').replace(' Mix', '')

def get_hex(name):
    if not isinstance(name, str): return '#95a5a6'
    if name in color_map: return color_map[name]
    lower = name.lower()
    if 'white' in lower and '/' in lower: return '#E8E8E8'
    if 'black' in lower: return '#333333'
    if 'brown' in lower: return '#8B4513'
    if 'blue' in lower: return '#778899'
    if 'orange' in lower: return '#FFA500'
    if 'gray' in lower: return '#808080'
    if 'yellow' in lower: return '#FFD700'
    if 'tan' in lower: return '#D2B48C'
    return '#95a5a6'

def get_luminance(hex_color):
    try:
        rgb = mcolors.hex2color(hex_color)
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    except: return 0.5

# --- PLOTTING LOGIC HELPERS ---

def plot_spectrum_subplot(ax, df_subset, animal_name, show_all=False, selected_colors=[]):
    if df_subset.empty: ax.axis('off'); return
    
    # סינון: אם המשתמש בחר צבעים ספציפיים, או הכל, או ברירת מחדל של הטופ 30
    if show_all:
        color_counts = df_subset['color'].value_counts()
    elif selected_colors: 
        df_subset = df_subset[df_subset['color'].isin(selected_colors)]
        color_counts = df_subset['color'].value_counts()
    else: 
        # ברירת מחדל: 30 הצבעים הנפוצים ביותר
        color_counts = df_subset['color'].value_counts().head(30)
    
    if color_counts.empty: ax.axis('off'); return
    
    df_colors = color_counts.reset_index()
    df_colors.columns = ['original_name', 'count']
    
    # חישובים
    df_colors['hex'] = df_colors['original_name'].apply(get_hex)
    df_colors['luminance'] = df_colors['hex'].apply(get_luminance)
    # שימוש בשמות היפים לתצוגה
    df_colors['display_name'] = df_colors['original_name'].apply(clean_label)
    
    # מיון לפי בהירות
    df_colors = df_colors.sort_values(by='luminance', ascending=True)
    
    # ציור הגרף עם השמות היפים בציר ה-Y
    bars = ax.barh(df_colors['display_name'], df_colors['count'], color=df_colors['hex'], edgecolor='gray', height=0.7)
    
    ax.set_title(f'{animal_name}', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # הסרת הקו השמאלי כמו בדוגמה החדשה
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    for bar, count in zip(bars, df_colors['count']):
        ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height()/2, f'{count:,}', va='center', fontsize=9, fontweight='bold', color='#444')

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

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1548199973-03cce0bbc87b?q=80&w=400&auto=format&fit=crop", use_container_width=True)
    st.title("🐾 Controls")
    min_year, max_year = int(df['intake_year'].min()), int(df['intake_year'].max())
    selected_years = st.slider("📆 Time Period:", min_year, max_year, (min_year, max_year))
    all_animals = df['animal_type'].unique()
    selected_animals = st.multiselect("🐶 Filter Animals:", all_animals, default=all_animals)
    st.markdown("---")
    page = st.radio("📚 Navigation:", [
        "1. Daily Activity Patterns",
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
filtered_df = df[global_mask].copy()

if filtered_df.empty:
    st.warning("⚠️ No data available for these filters. Please adjust the sidebar.")
    st.stop()

# --- MAIN PAGE HEADER ---
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
# PAGE 1: DAILY ACTIVITY (HEARTBEAT)
# ==========================================
if "1." in page:
    st.markdown("<div class='chart-header'>⏱️ Daily Activity Patterns</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='chart-explanation'>
    This chart visualizes the <b>Pulse of the Shelter</b> throughout the day.
    <br>• <b>Blue Line (Intake):</b> When animals arrive at the shelter (peaks in the morning).
    <br>• <b>Orange Line (Outcome):</b> When animals leave (adoption/transfer), peaking in the afternoon.
    <br>• The bottom bar shows the <b>Net Flow</b>: Blue bars mean the shelter is filling up, Orange bars mean it's emptying out.
    </div>
    """, unsafe_allow_html=True)

    filtered_df['Intake_Hour'] = filtered_df['intake_datetime'].dt.hour
    filtered_df['Outcome_Hour'] = filtered_df['outcome_datetime'].dt.hour
    
    hours = np.arange(6, 23)
    
    intake_counts = filtered_df['Intake_Hour'].value_counts().reindex(hours, fill_value=0)
    outcome_counts = filtered_df['Outcome_Hour'].value_counts().reindex(hours, fill_value=0)
    
    intake_total = intake_counts.sum()
    outcome_total = outcome_counts.sum()
    
    intake_pct = intake_counts / intake_total if intake_total > 0 else intake_counts * 0
    outcome_pct = outcome_counts / outcome_total if outcome_total > 0 else outcome_counts * 0
    diff = intake_pct - outcome_pct

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.patch.set_facecolor('#FDFBF7')

    for ax in axes:
        ax.set_facecolor('#FDFBF7')
        ax.axvspan(6, 10, color='gray', alpha=0.06)
        ax.axvspan(10, 16, color='gray', alpha=0.10)
        ax.axvspan(16, 19, color='orange', alpha=0.07)
        ax.axvspan(19, 22, color='gray', alpha=0.05)

    axes[0].fill_between(hours, intake_pct, color='#3498db', alpha=0.35, label='Intake (Arrivals)')
    axes[0].plot(hours, intake_pct, color='#1f5fa5', linewidth=2.5)
    axes[0].fill_between(hours, outcome_pct, color='#e67e22', alpha=0.35, label='Outcome (Departures)')
    axes[0].plot(hours, outcome_pct, color='#c4410e', linewidth=2.5)
    
    axes[0].set_title("The Shelter's Heartbeat: When Arrivals and Departures Don’t Align", fontsize=18, fontweight='bold', loc='left', pad=15, color='#2c3e50')
    axes[0].set_ylabel('Share of Daily Activity', fontweight='bold', color='#555')
    axes[0].legend(frameon=False, loc='upper right')
    axes[0].grid(axis='x', linestyle=':', alpha=0.4)
    sns.despine(ax=axes[0])
    
    y_limit = axes[0].get_ylim()[1]
    axes[0].text(8, y_limit*0.92, 'Morning Intake', ha='center', fontweight='bold', color='#1f5fa5')
    axes[0].text(13, y_limit*0.92, 'Midday Overlap', ha='center', fontweight='bold', color='#555555')
    axes[0].text(17.5, y_limit*0.92, 'Adoption Window', ha='center', fontweight='bold', color='#c4410e')

    axes[1].axhline(0, color='#222222', linewidth=1)
    axes[1].bar(hours, diff, color=['#3498db' if d > 0 else '#e67e22' for d in diff], alpha=0.85)
    axes[1].set_ylabel('Net Flow\n(Intake − Outcome)', fontweight='bold', color='#555')
    axes[1].set_xlabel('Hour of Day', fontweight='bold', color='#555')
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine(ax=axes[1])
    
    major_hours = np.arange(6, 23, 2)
    axes[1].set_xticks(major_hours)
    axes[1].set_xticklabels([f'{h:02d}:00' for h in major_hours], fontsize=11, fontweight='bold')
    axes[1].set_xticks(hours, minor=True)
    axes[1].tick_params(axis='x', which='minor', length=4, color='#999999')
    
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# PAGE 2: DEMOGRAPHICS (UPDATED WITH NEW LOGIC)
# ==========================================
elif "2." in page:
    st.markdown("<div class='chart-header'>🎨 Color Spectrum Analysis</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='chart-explanation'>
    <b>Understanding Coat Colors:</b><br>
    This visualization breaks down the animals based on their primary coat colors using popular display names (e.g., "Tuxedo", "Calico"). 
    Colors are sorted by <b>brightness</b> (darkest at the bottom, lightest at the top) to create a natural visual gradient.
    </div>
    """, unsafe_allow_html=True)

    col_sel, col_check = st.columns([3, 1])
    with col_sel:
        all_colors = sorted(filtered_df['color'].unique().astype(str))
        chosen_colors = st.multiselect("Filter Specific Colors (by original name):", all_colors)
    with col_check:
        st.write("")
        st.write("")
        show_all = st.checkbox("Show ALL Colors", value=False)
    
    temp_dog = filtered_df[filtered_df['animal_type'] == 'Dog']
    temp_cat = filtered_df[filtered_df['animal_type'] == 'Cat']
    
    # חישוב גובה הגרף באופן דינמי
    max_bars = 30 # ברירת מחדל
    if show_all:
        max_bars = max(len(temp_dog['color'].unique()), len(temp_cat['color'].unique()))
    elif chosen_colors:
        max_bars = len(chosen_colors)
    
    fig_height = max(6, max_bars * 0.4)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, fig_height))
    fig.patch.set_facecolor('#FDFBF7')
    
    if "Dog" in selected_animals: 
        plot_spectrum_subplot(axes[0], temp_dog, 'Dogs', show_all, chosen_colors)
    else: 
        axes[0].axis('off')
        
    if "Cat" in selected_animals: 
        plot_spectrum_subplot(axes[1], temp_cat, 'Cats', show_all, chosen_colors)
    else: 
        axes[1].axis('off')
        
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# PAGE 3: SEASONALITY
# ==========================================
elif "3." in page:
    st.markdown("<div class='chart-header'>📅 Seasonal Intake Patterns</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='chart-explanation'>
    <b>How to read these charts:</b><br>
    1. <b>Daily Heatmap:</b> Shows intensity of intakes per day of the week over the year. Darker red means more animals arrived on that specific day.
    2. <b>Monthly Cycle:</b> A polar (circular) chart showing the "seasons" of the shelter. It highlights which months have the highest volume (usually summer for cats).
    </div>
    """, unsafe_allow_html=True)

    valid_years = sorted(filtered_df['intake_year'].unique())
    target_year = st.selectbox("Select Focus Year:", valid_years, index=0)
    st.subheader(f"Daily Heatmap ({target_year})")
    
    matrix_cats, max_cat = get_cal_matrix(filtered_df[filtered_df['animal_type'] == 'Cat'], target_year)
    matrix_dogs, max_dog = get_cal_matrix(filtered_df[filtered_df['animal_type'] == 'Dog'], target_year)
    global_max = max(max_cat, max_dog)
    
    fig_cal, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig_cal.patch.set_facecolor('#FDFBF7')
    heatmap_cmap = 'OrRd'
    
    days_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    
    if "Cat" in selected_animals:
        sns.heatmap(matrix_cats, ax=axes[0], cmap=heatmap_cmap, linewidths=0.5, linecolor='white', square=True,
                    vmin=0, vmax=global_max, cbar=False)
        axes[0].set_title('🐱 Cats Intake', fontweight='bold', fontsize=12, loc='left')
        axes[0].set_xticks([])
        axes[0].set_yticklabels(days_labels, rotation=0, fontsize=9)
    else: axes[0].axis('off')
    
    if "Dog" in selected_animals:
        sns.heatmap(matrix_dogs, ax=axes[1], cmap=heatmap_cmap, linewidths=0.5, linecolor='white', square=True,
                    vmin=0, vmax=global_max, cbar=False)
        axes[1].set_title('🐶 Dogs Intake', fontweight='bold', fontsize=12, loc='left')
        axes[1].set_yticklabels(days_labels, rotation=0, fontsize=9)
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
    
    # 4a. Puppy Surrenders
    st.subheader("🥀 The 'Season of Regret'")
    st.caption("Puppy surrenders peak in Jan/Feb and drop in Summer")
    
    if "Dog" in selected_animals:
        df_surrender = filtered_df[(filtered_df['animal_type'] == 'Dog') & (filtered_df['intake_type'] == 'Owner Surrender')].copy()
        puppy_surrenders = df_surrender[df_surrender['age_group'].isin(['Puppy', 'Puppy_Kitten', 'Kitten'])]
        
        if not puppy_surrenders.empty:
            counts = puppy_surrenders.groupby(puppy_surrenders['intake_datetime'].dt.month).size().reindex(range(1, 13), fill_value=0)
            months = np.arange(1, 13)
            values = counts.values
            max_val = values.max(); min_val = values.min() if values.min() > 0 else 1
            max_month_idx = values.argmax() + 1; min_month_idx = values.argmin() + 1
            avg_val = values.mean(); ratio = max_val / min_val
            
            # Increased Figure Size
            fig_area, ax = plt.subplots(figsize=(15, 10))
            fig_area.patch.set_facecolor('#FDFBF7')
            
            ax.plot(months, values, color='#c0392b', linewidth=2.5, zorder=3)
            ax.fill_between(months, values, color='#e74c3c', alpha=0.2)
            winter_mask = (months <= 2) | (months >= 11)
            ax.fill_between(months, values, where=winter_mask, color='#c0392b', alpha=0.3, interpolate=True)
            ax.axhline(y=avg_val, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            
            ax.annotate(f'The "Regret" Peak\n(Jan: {max_val})', 
                        xy=(max_month_idx, max_val), xytext=(max_month_idx + 1.5, max_val + 2),
                        arrowprops=dict(facecolor='#c0392b', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=10, fontweight='bold', color='#c0392b')

            ax.annotate(f'Summer Low\n({calendar.month_abbr[min_month_idx]}: {min_val})', 
                        xy=(min_month_idx, min_val), xytext=(min_month_idx - 1, min_val + 8),
                        arrowprops=dict(facecolor='#2980b9', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=10, fontweight='bold', color='#2980b9', ha='center')

            text_content = f"Puppy surrenders are\n{ratio:.1f}x higher in Winter!"
            ax.text(6.5, max_val * 0.9, text_content, 
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9),
                    ha='center', fontsize=11, fontweight='bold', color='#2c3e50')

            ax.set_xticks(months)
            ax.set_xticklabels([calendar.month_abbr[i] for i in months], fontsize=10)
            ax.set_ylim(0, max_val * 1.25)
            ax.set_ylabel("Number of Surrendered Puppies")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig_area)
        else: st.info("Not enough puppy surrender data.")
    else: st.warning("Select 'Dogs' in the sidebar to view this chart.")

    st.markdown("---")

    # 4b. Adoption Rates
    st.subheader("🏠 Adoption Success")
    st.caption("Adoption rates by age group")
    
    # 1. מיפוי לקבוצות מאוחדות
    age_map = {
        'Puppy': 'Puppy or Kitten',
        'Kitten': 'Puppy or Kitten',
        'Puppy_Kitten': 'Puppy or Kitten',
        'Junior': 'Juvenile',
        'Adult': 'Adult',
        'Senior': 'Senior'
    }
    
    filtered_df['plot_age_group'] = filtered_df['age_group'].map(age_map).fillna(filtered_df['age_group'])

    # 2. סידור
    desired_order = ['Puppy or Kitten', 'Juvenile', 'Adult', 'Senior']
    filtered_df['plot_age_group'] = pd.Categorical(
        filtered_df['plot_age_group'], 
        categories=desired_order, 
        ordered=True
    )
    
    # 3. חישוב
    age_compares = filtered_df.groupby(['animal_type', 'plot_age_group'], observed=False)['outcome_type'] \
        .apply(lambda x: (x == 'Adopted').mean()).reset_index(name='Adopted')
    
    # 4. ציור (צבעים מקוריים)
    fig_rates, ax_rates = plt.subplots(1, 2, figsize=(15, 6))
    fig_rates.patch.set_facecolor('#FDFBF7')
    
    color_adopted = "#4ECDC4"   # Turquoise
    color_other = "#fab1a0"     # Salmon/Peach
    
    for i, animal in enumerate(['Cat', 'Dog']):
        if animal in selected_animals:
            subset = age_compares[age_compares['animal_type'] == animal].sort_values('plot_age_group')
            subset = subset.dropna(subset=['Adopted'])
            
            if not subset.empty:
                x = subset["plot_age_group"]
                adopted_vals = subset["Adopted"] * 100
                not_adopted_vals = (1 - subset["Adopted"]) * 100
                
                ax_rates[i].bar(x, adopted_vals, color=color_adopted, label="Adopted", 
                              width=0.65, edgecolor='white', linewidth=1.5)
                ax_rates[i].bar(x, not_adopted_vals, bottom=adopted_vals, color=color_other, label="Other", 
                              width=0.65, edgecolor='white', linewidth=1.5)
                
                ax_rates[i].set_title(f"{animal}s", fontweight='bold', fontsize=14, color='#333')
                ax_rates[i].set_ylim(0, 100)
                ax_rates[i].set_ylabel("Percentage" if i==0 else "")
                
                for idx, (a, n) in enumerate(zip(adopted_vals, not_adopted_vals)):
                    if a > 5:
                        ax_rates[i].text(idx, a/2, f"{a:.1f}%", ha='center', va='center', 
                                       color='white', fontweight='bold', fontsize=10)
                    if n > 5:
                        ax_rates[i].text(idx, a + n/2, f"{n:.1f}%", ha='center', va='center', 
                                       color='white', fontweight='bold', fontsize=10)

                ax_rates[i].spines['top'].set_visible(False)
                ax_rates[i].spines['right'].set_visible(False)
                ax_rates[i].grid(axis='y', linestyle='--', alpha=0.3)
                
                if i == 1:
                    ax_rates[i].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        else:
            ax_rates[i].axis('off')
            
    st.pyplot(fig_rates)
