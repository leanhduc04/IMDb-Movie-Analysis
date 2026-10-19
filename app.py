import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Style chung cho biểu đồ
sns.set_theme(
    style="whitegrid",
    palette="muted"
)

plt.rcParams.update({
    "figure.figsize": (11, 6),
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

st.set_page_config(
    page_title="IMDb Movie Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

/* Giảm khoảng trống phía trên */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Tiêu đề chính */
.main-title {
    font-size: 2.6rem;
    font-weight: 700;
    margin-bottom: 0;
    text-align: center;
}

/* Mô tả dưới tiêu đề */
.subtitle {
    font-size: 1rem;
    opacity: 0.7;
    margin-top: 0.3rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

/* Metric */
[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.25);
    padding: 15px;
    border-radius: 10px;
}

/* Số trong metric */
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 600;
}

/* Tab */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}

/* Chữ trong tab */
.stTabs [data-baseweb="tab"] {
    padding-left: 14px;
    padding-right: 14px;
    font-weight: 500;
}

/* Tab đang được chọn */
.stTabs [aria-selected="true"] {
    font-weight: 700;
}

/* Nội dung tab */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

</style>
""", unsafe_allow_html=True)

# Đọc dữ liệu đã làm sạch
@st.cache_data
def load_data():
    return pd.read_csv("data/imdb_cleaned.csv")

df = load_data()

# Tải mô hình Linear Regression đã huấn luyện
@st.cache_resource
def load_model():
    return joblib.load("linear_revenue_model.pkl")

linear_model = load_model()

# Tiêu đề
st.markdown(
    '<div class="main-title">🎬 IMDb Movie Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Phân tích và trực quan hóa dữ liệu phim IMDb'
    '</div>',
    unsafe_allow_html=True
)

# Các chỉ số tổng quan
total_movies = len(df)
avg_rating = df["Rating"].mean()
avg_revenue = df["Revenue"].mean()
total_votes = df["Votes"].sum()

st.subheader("Tổng quan dữ liệu")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Số lượng phim",
        value=f"{total_movies:,}"
    )

with col2:
    st.metric(
        label="Rating trung bình",
        value=f"{avg_rating:.2f} / 10"
    )

with col3:
    st.metric(
        label="Doanh thu trung bình",
        value=f"${avg_revenue:.2f}M"
    )

with col4:
    st.metric(
        label="Tổng lượt bình chọn",
        value=f"{total_votes / 1_000_000:.1f}M"
    )

# BỘ LỌC
st.sidebar.header("Bộ lọc dữ liệu")
st.sidebar.caption(
    "Tùy chỉnh dữ liệu hiển thị trên dashboard"
)

# Lọc theo năm
years = sorted(df["Year"].dropna().unique())

selected_year = st.sidebar.selectbox(
    "Chọn năm",
    ["Tất cả"] + list(years)
)

# Lọc theo thể loại
genres = sorted(
    df["Genre"]
    .str.split(",")
    .explode()
    .str.strip()
    .dropna()
    .unique()
)

selected_genre = st.sidebar.selectbox(
    "Chọn thể loại",
    ["Tất cả"] + list(genres)
)

# Lọc theo Rating
min_rating = st.sidebar.slider(
    "Rating tối thiểu",
    min_value=float(df["Rating"].min()),
    max_value=float(df["Rating"].max()),
    value=float(df["Rating"].min()),
    step=0.1
)

# ÁP DỤNG BỘ LỌC
filtered_df = df.copy()

if selected_year != "Tất cả":
    filtered_df = filtered_df[
        filtered_df["Year"] == selected_year
    ]

if selected_genre != "Tất cả":
    filtered_df = filtered_df[
        filtered_df["Genre"].apply(
            lambda x: selected_genre in [
                genre.strip()
                for genre in x.split(",")
            ]
            if pd.notna(x)
            else False
        )
    ]

filtered_df = filtered_df[
    filtered_df["Rating"] >= min_rating
]
has_data = not filtered_df.empty

st.sidebar.divider()

st.sidebar.write(
    f"**Số phim phù hợp:** {len(filtered_df):,}"
)

st.sidebar.caption(
    f"Trên tổng số {len(df):,} phim"
)

st.subheader("Phân tích dữ liệu")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Phân bố",
    "📈 Xu hướng",
    "🎭 Thể loại",
    "🏆 Bảng xếp hạng",
    "🔗 Tương quan",
    "🔎 Tra cứu",
    "🧠 Dự đoán doanh thu",
    "📋 Dữ liệu"
])

with tab1:
    st.subheader("Phân tích phân bố")

    if not has_data:
        st.warning(
            "Không có dữ liệu phù hợp với bộ lọc hiện tại."
        )
    else:
        variable_labels = {
            "Runtime": "Thời lượng phim",
            "Rating": "IMDb Rating",
            "Votes": "Lượt bình chọn",
            "Revenue": "Doanh thu",
            "Metascore": "Metascore"
        }
        
        selected_variable = st.selectbox(
            "Chọn biến cần phân tích",
            options=list(variable_labels.keys()),
            format_func=lambda x: variable_labels[x],
            key="univariate_variable"
        )
        
        data = filtered_df[selected_variable].dropna()
    
        if data.empty:
            st.warning("Không có dữ liệu hợp lệ cho biến được chọn.")
        else:
            # THỐNG KÊ MÔ TẢ
            st.write("### Thống kê mô tả")
    
            col1, col2, col3, col4 = st.columns(4)
        
            with col1:
                st.metric(
                    "Trung bình",
                    f"{data.mean():,.2f}"
                )
        
            with col2:
                st.metric(
                    "Trung vị",
                    f"{data.median():,.2f}"
                )
        
            with col3:
                st.metric(
                    "Độ lệch chuẩn",
                    f"{data.std():,.2f}"
                )
        
            with col4:
                st.metric(
                    "Độ lệch",
                    f"{data.skew():.2f}"
                )
        
            col5, col6, col7, col8 = st.columns(4)
        
            with col5:
                st.metric(
                    "Nhỏ nhất",
                    f"{data.min():,.2f}"
                )
        
            with col6:
                st.metric(
                    "Q1 (25%)",
                    f"{data.quantile(0.25):,.2f}"
                )
        
            with col7:
                st.metric(
                    "Q3 (75%)",
                    f"{data.quantile(0.75):,.2f}"
                )
        
            with col8:
                st.metric(
                    "Lớn nhất",
                    f"{data.max():,.2f}"
                )
        
            st.divider()
        
            # HISTOGRAM + KDE
            st.write(
                f"### Phân bố của {selected_variable}"
            )
        
            fig, ax = plt.subplots(
                figsize=(11, 5)
            )
        
            sns.histplot(
                data,
                bins=30,
                kde=True,
                color="royalblue",
                ax=ax
            )
        
            ax.set_xlabel(selected_variable)
            ax.set_ylabel("Tần suất")
        
            plt.tight_layout()
        
            st.pyplot(fig)
            plt.close(fig)
        
            st.divider()
    
            # BOXPLOT
            st.write(
                f"### Boxplot của {selected_variable}"
            )
        
            fig, ax = plt.subplots(
                figsize=(11, 3)
            )
        
            sns.boxplot(
                x=data,
                color="coral",
                ax=ax
            )
        
            ax.set_xlabel(selected_variable)
        
            plt.tight_layout()
        
            st.pyplot(fig)
            plt.close(fig)

with tab2:
    st.subheader("Phân tích xu hướng theo thời gian")

    if not has_data:
        st.warning(
            "Không có dữ liệu phù hợp với bộ lọc hiện tại."
        )
    else:
        # 1. SỐ LƯỢNG PHIM THEO NĂM
        st.write("### Số lượng phim phát hành theo năm")
    
        movies_by_year = (
            filtered_df
            .groupby("Year")
            .size()
            .sort_index()
        )
    
        fig, ax = plt.subplots(figsize=(11, 5))
    
        bars = ax.bar(
            movies_by_year.index,
            movies_by_year.values,
            color="steelblue",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85
        )

        ax.grid(axis="y", alpha=0.3)
        ax.grid(False, axis="x")
    
        ax.set_xlabel("Năm")
        ax.set_ylabel("Số lượng phim")
    
        # Hiển thị năm không có dấu phẩy
        ax.set_xticks(movies_by_year.index)
        ax.set_xticklabels(
            [str(int(year)) for year in movies_by_year.index]
        )
    
        # Hiển thị số lượng phim trên từng điểm
        for year, value in zip(
            movies_by_year.index,
            movies_by_year.values
        ):
            ax.annotate(
                str(value),
                (year, value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center"
            )
    
        ax.grid(
            axis="y",
            alpha=0.3
        )
    
        plt.tight_layout()
    
        st.pyplot(fig)
        plt.close(fig)
    
        st.divider()
    
        # 2. DOANH THU THEO NĂM
        st.write("### Doanh thu theo năm")
    
        revenue_by_year = (
            filtered_df
            .groupby("Year")["Revenue"]
            .agg(
                Total_Revenue="sum",
                Median_Revenue="median"
            )
            .sort_index()
        )
    
        fig, ax1 = plt.subplots(
            figsize=(11, 6)
        )
    
        # Tổng doanh thu
        line1 = ax1.plot(
            revenue_by_year.index,
            revenue_by_year["Total_Revenue"],
            marker="o",
            linewidth=2,
            color="#2e7d32",
            label="Tổng doanh thu"
        )
    
        ax1.set_xlabel("Năm")
    
        ax1.set_ylabel(
            "Tổng doanh thu (triệu USD)"
        )
    
        ax1.set_xticks(
            revenue_by_year.index
        )
    
        ax1.set_xticklabels(
            [
                str(int(year))
                for year in revenue_by_year.index
            ]
        )
    
        ax1.grid(
            axis="y",
            alpha=0.3
        )
    
        # Trục Y thứ hai cho Median
        ax2 = ax1.twinx()
    
        line2 = ax2.plot(
            revenue_by_year.index,
            revenue_by_year["Median_Revenue"],
            marker="s",
            linestyle="--",
            linewidth=2,
            color="#e65100",
            label="Trung vị doanh thu"
        )
    
        ax2.set_ylabel(
            "Trung vị doanh thu (triệu USD)"
        )
    
        ax2.grid(False)
    
        # Hiển thị giá trị Total Revenue
        for year, value in zip(
            revenue_by_year.index,
            revenue_by_year["Total_Revenue"]
        ):
            ax1.annotate(
                f"{value:,.0f}",
                (year, value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8
            )
    
        # Hiển thị giá trị Median Revenue
        for year, value in zip(
            revenue_by_year.index,
            revenue_by_year["Median_Revenue"]
        ):
            ax2.annotate(
                f"{value:.1f}",
                (year, value),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=8
            )
    
        # Gộp legend của 2 trục
        lines = line1 + line2
    
        labels = [
            line.get_label()
            for line in lines
        ]
    
        ax1.legend(
            lines,
            labels,
            loc="upper left"
        )
    
        plt.tight_layout()
    
        st.pyplot(fig)
        plt.close(fig)
    
        st.divider()
    
        # 3. RATING VÀ METASCORE
        st.write("### Điểm đánh giá trung bình theo năm")
    
        score_by_year = (
            filtered_df
            .groupby("Year")
            .agg(
                Mean_Rating=("Rating", "mean"),
                Mean_Metascore=("Metascore", "mean")
            )
            .sort_index()
        )
    
        # Metascore: 0-100
        # Quy đổi về 0-10 để so sánh với Rating
        score_by_year["Scaled_Metascore"] = (
            score_by_year["Mean_Metascore"] / 10
        )
    
        fig, ax = plt.subplots(
            figsize=(11, 5.5)
        )
    
        # Rating
        ax.plot(
            score_by_year.index,
            score_by_year["Mean_Rating"],
            marker="o",
            linewidth=2,
            color="#1976d2",
            label="IMDb Rating trung bình"
        )
    
        # Metascore đã quy đổi
        ax.plot(
            score_by_year.index,
            score_by_year["Scaled_Metascore"],
            marker="s",
            linestyle="--",
            linewidth=2,
            color="#d32f2f",
            label="Metascore trung bình (thang 10)"
        )
    
        ax.set_xlabel("Năm")
    
        ax.set_ylabel(
            "Điểm trung bình (thang 10)"
        )
    
        ax.set_xticks(
            score_by_year.index
        )
    
        ax.set_xticklabels(
            [
                str(int(year))
                for year in score_by_year.index
            ]
        )
    
        # Hiển thị Rating
        for year, value in zip(
            score_by_year.index,
            score_by_year["Mean_Rating"]
        ):
            ax.annotate(
                f"{value:.2f}",
                (year, value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8
            )
    
        # Hiển thị Metascore
        for year, value in zip(
            score_by_year.index,
            score_by_year["Scaled_Metascore"]
        ):
            ax.annotate(
                f"{value:.2f}",
                (year, value),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=8
            )
    
        ax.legend()
    
        ax.grid(
            axis="y",
            alpha=0.3
        )
    
        plt.tight_layout()
    
        st.pyplot(fig)
        plt.close(fig)

with tab3:
    st.subheader("Phân tích theo thể loại")
    if not has_data:
        st.warning(
            "Không có dữ liệu phù hợp với bộ lọc hiện tại."
        )
    else:
        # CHUẨN BỊ DỮ LIỆU GENRE
        genre_df = filtered_df.copy()

        genre_df["Genre"] = (
            genre_df["Genre"]
            .str.split(",")
        )
    
        genre_df = genre_df.explode("Genre")
    
        genre_df["Genre"] = (
            genre_df["Genre"]
            .str.strip()
        )

        genre_df = genre_df[
            genre_df["Genre"].notna()
            & (genre_df["Genre"] != "")
        ]
    

        # 1. MỨC ĐỘ PHỔ BIẾN CỦA CÁC THỂ LOẠI
        st.write("### Mức độ phổ biến của các thể loại")
    
        genre_counts = (
            genre_df["Genre"]
            .value_counts()
            .sort_values(ascending=True)
        )
    
        fig, ax = plt.subplots(figsize=(11, 7))
    
        colors = sns.color_palette(
            "Blues_r",
            n_colors=len(genre_counts)
        )[::-1]
        
        bars = ax.barh(
            genre_counts.index,
            genre_counts.values,
            color=colors,
            edgecolor="black",
            linewidth=0.6
        )
    
        ax.set_xlabel("Số lượng phim")
        ax.set_ylabel("Thể loại")
    
        # Hiển thị số lượng phim trên từng thanh
        for bar in bars:
            width = bar.get_width()
    
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f" {int(width)}",
                va="center"
            )
    
        ax.grid(
            axis="x",
            alpha=0.3
        )
    
        plt.tight_layout()
    
        st.pyplot(fig)
        plt.close(fig)
    
        st.divider()
    
        # 2. DOANH THU THEO THỂ LOẠI
        st.write("### Doanh thu theo thể loại")
    
        genre_revenue = (
            genre_df
            .groupby("Genre")
            .agg(
                Mean_Revenue=("Revenue", "mean"),
                Median_Revenue=("Revenue", "median"),
            )
            .sort_values(
                "Median_Revenue",
                ascending=True
            )
        )
    
        fig, ax = plt.subplots(figsize=(11, 8))
    
        y = np.arange(
            len(genre_revenue)
        )
    
        height = 0.38
    
        # Doanh thu trung bình
        ax.barh(
            y - height / 2,
            genre_revenue["Mean_Revenue"],
            height=height,
            color="#1976d2",
            edgecolor="black",
            linewidth=0.5,
            label="Doanh thu trung bình"
        )
        
        ax.barh(
            y + height / 2,
            genre_revenue["Median_Revenue"],
            height=height,
            color="#f57c00",
            edgecolor="black",
            linewidth=0.5,
            label="Trung vị doanh thu"
        )
    
        ax.set_yticks(y)
    
        ax.set_yticklabels(
            genre_revenue.index
        )
    
        ax.set_xlabel(
            "Doanh thu (triệu USD)"
        )
    
        ax.set_ylabel("Thể loại")
    
        ax.legend()
    
        ax.grid(
            axis="x",
            alpha=0.3
        )
    
        plt.tight_layout()
    
        st.pyplot(fig)
        plt.close(fig)
    
        st.divider()
    
        # 3. RATING VÀ METASCORE THEO THỂ LOẠI
        st.write("### Điểm đánh giá trung bình theo thể loại")
    
        genre_scores = (
            genre_df
            .groupby("Genre")
            .agg(
                Mean_Rating=("Rating", "mean"),
                Mean_Metascore=("Metascore", "mean")
            )
        )
    
        # Metascore có thang 0-100
        # Quy đổi về thang 0-10
        genre_scores["Scaled_Metascore"] = (
            genre_scores["Mean_Metascore"] / 10
        )
    
        # Sắp xếp theo Rating
        genre_scores = (
            genre_scores
            .sort_values(
                "Mean_Rating",
                ascending=True
            )
        )
    
        fig, ax = plt.subplots(figsize=(11, 8))
    
        y = np.arange(
            len(genre_scores)
        )
    
        height = 0.38
    
        # Rating
        ax.barh(
            y - height / 2,
            genre_scores["Mean_Rating"],
            height=height,
            color="#1565c0",
            edgecolor="black",
            linewidth=0.5,
            label="IMDb Rating trung bình"
        )
    
        # Metascore
        ax.barh(
            y + height / 2,
            genre_scores["Scaled_Metascore"],
            height=height,
            color="#c62828",
            edgecolor="black",
            linewidth=0.5,
            label="Metascore trung bình (thang 10)"
        )

        ax.axvline(
            x=7.0,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label="Mốc 7.0 điểm"
        )
    
        ax.set_yticks(y)
    
        ax.set_yticklabels(
            genre_scores.index
        )
    
        ax.set_xlabel(
            "Điểm trung bình (thang 10)"
        )
    
        ax.set_ylabel("Thể loại")
    
        ax.set_xlim(0, 10)
    
        ax.legend()
    
        ax.grid(
            axis="x",
            alpha=0.3
        )
    
        plt.tight_layout()
    
        st.pyplot(fig)
        plt.close(fig)

with tab4:
    st.subheader("Bảng xếp hạng")
    if not has_data:
        st.warning(
            "Không có dữ liệu phù hợp với bộ lọc hiện tại."
        )
    else:
        # 1. Top 10 phim theo doanh thu
        st.write("### Top 10 phim theo doanh thu")
    
        top_revenue = (
            filtered_df
            .dropna(subset=["Revenue"])
            .sort_values(
                "Revenue",
                ascending=False
            )
            .head(10)
            .sort_values(
                "Revenue",
                ascending=True
            )
        )
    
        fig, ax = plt.subplots(figsize=(11, 6))

        colors = sns.color_palette(
            "crest_r",
            n_colors=len(top_revenue)
        )[::-1]
        
        bars = ax.barh(
            top_revenue["Title"],
            top_revenue["Revenue"],
            color=colors,
            edgecolor="black",
            linewidth=0.6
        )
    
        ax.set_xlabel("Doanh thu (triệu USD)")
        ax.set_ylabel("Tên phim")
    
        # Hiển thị doanh thu trên từng thanh
        for bar in bars:
            width = bar.get_width()
    
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f" {width:.1f}",
                va="center"
            )
    
        ax.grid(
            axis="x",
            alpha=0.3
        )
    
        plt.tight_layout()
    
        st.pyplot(fig)
        plt.close(fig)
    
        st.divider()
    
    
        # 2. Top 10 phim theo Rating
        st.write("### Top 10 phim theo IMDb Rating")
    
        # Nếu Rating bằng nhau,
        # ưu tiên phim có Votes cao hơn
        top_rating = (
            filtered_df
            .dropna(subset=["Rating", "Votes"])
            .sort_values(
                ["Rating", "Votes"],
                ascending=[False, False]
            )
            .head(10)
            .sort_values(
                ["Rating", "Votes"],
                ascending=[True, True]
            )
        )
    
        fig, ax = plt.subplots(figsize=(11, 6))
    
        colors = sns.color_palette(
            "flare_r",
            n_colors=len(top_rating)
        )[::-1]
        
        bars = ax.barh(
            top_rating["Title"],
            top_rating["Rating"],
            color=colors,
            edgecolor="black",
            linewidth=0.6
        )
    
        ax.set_xlabel("IMDb Rating")
        ax.set_ylabel("Tên phim")
    
        ax.set_xlim(0, 10)
    
        # Hiển thị Rating
        for bar in bars:
            width = bar.get_width()
    
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f" {width:.1f}",
                va="center"
            )
    
        ax.grid(
            axis="x",
            alpha=0.3
        )
    
        plt.tight_layout()
    
        st.pyplot(fig)
        plt.close(fig)
    
        st.divider()
    
    
        # 3. Top 10 đạo diễn theo tổng doanh thu
        st.write("### Top 10 đạo diễn theo tổng doanh thu")
    
        director_revenue = (
            filtered_df
            .dropna(subset=["Director", "Revenue"])
            .groupby("Director")
            .agg(
                Total_Revenue=("Revenue", "sum"),
                Movie_Count=("Title", "count")
            )
            .sort_values(
                "Total_Revenue",
                ascending=False
            )
            .head(10)
            .sort_values(
                "Total_Revenue",
                ascending=True
            )
        )
    
        fig, ax = plt.subplots(figsize=(11, 6))
    
        colors = sns.color_palette(
            "YlGnBu_r",
            n_colors=len(director_revenue)
        )[::-1]
        
        bars = ax.barh(
            director_revenue.index,
            director_revenue["Total_Revenue"],
            color=colors,
            edgecolor="black",
            linewidth=0.6
        )
    
        ax.set_xlabel(
            "Tổng doanh thu (triệu USD)"
        )
    
        ax.set_ylabel("Đạo diễn")
    
        # Hiển thị doanh thu và số phim
        for bar, (_, row) in zip(
            bars,
            director_revenue.iterrows()
        ):
            width = bar.get_width()
    
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f" {width:.1f} ({int(row['Movie_Count'])} phim)",
                va="center"
            )
    
        ax.grid(
            axis="x",
            alpha=0.3
        )
    
        plt.tight_layout()
    
        st.pyplot(fig)
        plt.close(fig)
    
        st.divider()
    
    
        # 4. Top 10 đạo diễn theo Rating trung bình
        st.write("### Top 10 đạo diễn theo Rating trung bình")
    
        st.caption(
            "Chỉ xét các đạo diễn có ít nhất 3 phim trong dữ liệu sau khi áp dụng bộ lọc."
        )
    
        director_rating = (
            filtered_df
            .dropna(subset=["Director", "Rating"])
            .groupby("Director")
            .agg(
                Mean_Rating=("Rating", "mean"),
                Movie_Count=("Title", "count")
            )
        )
    
        # Chỉ giữ đạo diễn có ít nhất 3 phim
        director_rating = director_rating[
            director_rating["Movie_Count"] >= 3
        ]
    
        director_rating = (
            director_rating
            .sort_values(
                "Mean_Rating",
                ascending=False
            )
            .head(10)
            .sort_values(
                "Mean_Rating",
                ascending=True
            )
        )
    
        fig, ax = plt.subplots(figsize=(11, 6))
    
        colors = sns.color_palette(
            "magma_r",
            n_colors=len(director_rating)
        )[::-1]
        
        bars = ax.barh(
            director_rating.index,
            director_rating["Mean_Rating"],
            color=colors,
            edgecolor="black",
            linewidth=0.6
        )
    
        ax.set_xlabel(
            "IMDb Rating trung bình"
        )
    
        ax.set_ylabel("Đạo diễn")
    
        ax.set_xlim(0, 10)
    
        # Hiển thị Rating và số phim
        for bar, (_, row) in zip(
            bars,
            director_rating.iterrows()
        ):
            width = bar.get_width()
    
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f" {width:.2f} ({int(row['Movie_Count'])} phim)",
                va="center"
            )
    
        ax.grid(
            axis="x",
            alpha=0.3
        )
    
        plt.tight_layout()
    
        st.pyplot(fig)
        plt.close(fig)

with tab5:
    st.subheader("Phân tích tương quan")
    if not has_data:
        st.warning(
            "Không có dữ liệu phù hợp với bộ lọc hiện tại."
        )
    else:
        # 1. MA TRẬN TƯƠNG QUAN PEARSON
        st.write("### Ma trận tương quan giữa các biến")
    
        numeric_features = [
            "Year",
            "Runtime",
            "Rating",
            "Votes",
            "Revenue",
            "Metascore"
        ]
    
        corr_data = filtered_df[numeric_features].copy()
    
        # Tính ma trận tương quan
        corr_matrix = corr_data.corr(method="pearson")
    
        # Tạo mask để chỉ hiển thị nửa dưới ma trận
        mask = np.triu(
            np.ones_like(
                corr_matrix,
                dtype=bool
            )
        )
    
        fig, ax = plt.subplots(figsize=(10, 7))
    
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=1,
            linecolor="white",
            cbar_kws={
                "label": "Hệ số tương quan Pearson (r)"
            },
            ax=ax
        )
    
        ax.set_title(
            "Ma trận tương quan giữa các biến định lượng"
        )
    
        plt.tight_layout()
    
        st.pyplot(fig)
        plt.close(fig)
    
        st.divider()
    
        # 2. VOTES VÀ REVENUE
        st.write(
            "### Mối quan hệ giữa Votes và Revenue"
        )
    
        votes_revenue = (
            filtered_df[
                ["Votes", "Revenue"]
            ]
            .dropna()
        )
    
        # Hệ số tương quan Pearson
        if len(votes_revenue) >= 2:
    
            r_votes_revenue = (
                votes_revenue["Votes"]
                .corr(
                    votes_revenue["Revenue"],
                    method="pearson"
                )
            )
    
            col1, col2 = st.columns(2)
    
            with col1:
                st.metric(
                    "Hệ số tương quan Pearson (r)",
                    f"{r_votes_revenue:.2f}"
                )
    
            with col2:
                st.metric(
                    "Số quan sát",
                    f"{len(votes_revenue):,}"
                )
    
            fig, ax = plt.subplots(
                figsize=(10, 6)
            )
    
            sns.regplot(
                data=votes_revenue,
                x="Votes",
                y="Revenue",
                color="#00695c",
                scatter_kws={
                    "alpha": 0.4,
                    "s": 30
                },
                line_kws={
                    "color": "#d32f2f",
                    "linewidth": 2
                },
                ax=ax
            )
    
            ax.set_xlabel(
                "Số lượt bình chọn (Votes)"
            )
    
            ax.set_ylabel(
                "Doanh thu (triệu USD)"
            )
    
            ax.set_title(
                f"Votes vs Revenue (r = {r_votes_revenue:.2f})"
            )
    
            ax.grid(
                alpha=0.3
            )
    
            plt.tight_layout()
    
            st.pyplot(fig)
            plt.close(fig)
    
        else:
            st.warning(
                "Không đủ dữ liệu để tính tương quan Votes và Revenue."
            )
    
        st.divider()
    
        # 3. METASCORE VÀ RATING
        st.write(
            "### Mối quan hệ giữa Metascore và IMDb Rating"
        )
    
        metascore_rating = (
            filtered_df[
                ["Metascore", "Rating"]
            ]
            .dropna()
        )
    
        if len(metascore_rating) >= 2:
            r_metascore_rating = (
                metascore_rating["Metascore"]
                .corr(
                    metascore_rating["Rating"],
                    method="pearson"
                )
            )
    
            col1, col2 = st.columns(2)
    
            with col1:
                st.metric(
                    "Hệ số tương quan Pearson (r)",
                    f"{r_metascore_rating:.2f}"
                )
    
            with col2:
                st.metric(
                    "Số quan sát",
                    f"{len(metascore_rating):,}"
                )
    
            fig, ax = plt.subplots(
                figsize=(10, 6)
            )
    
            sns.regplot(
            data=metascore_rating,
            x="Metascore",
            y="Rating",
            color="#4527a0",
            scatter_kws={
                "alpha": 0.4,
                "s": 30
            },
            line_kws={
                "color": "#f57c00",
                "linewidth": 2
            },
            ax=ax
        )
    
            ax.set_xlabel(
                "Metascore"
            )
    
            ax.set_ylabel(
                "IMDb Rating"
            )
    
            ax.set_title(
                f"Metascore vs IMDb Rating "
                f"(r = {r_metascore_rating:.2f})"
            )
    
            ax.grid(
                alpha=0.3
            )
    
            plt.tight_layout()
    
            st.pyplot(fig)
            plt.close(fig)
    
        else:
            st.warning(
                "Không đủ dữ liệu để tính tương quan Metascore và Rating."
            )

with tab6:
    st.subheader("Tra cứu phim")

    # TÌM KIẾM THEO TÊN
    search_title = st.text_input(
        "Nhập tên phim",
        placeholder="Ví dụ: Guardians, Avengers, Star Wars...",
        key="movie_search"
    )

    # Nếu nhập tên -> lọc danh sách phim
    if search_title.strip():

        movie_options = df[
            df["Title"].str.contains(
                search_title.strip(),
                case=False,
                na=False,
                regex=False
            )
        ].copy()

    # Nếu không nhập -> hiển thị toàn bộ phim
    else:
        movie_options = df.copy()

    # KIỂM TRA KẾT QUẢ
    if movie_options.empty:
        st.warning(
            "Không tìm thấy phim phù hợp với từ khóa."
        )

    else:
        # Sắp xếp tên phim
        movie_options = movie_options.sort_values(
            "Title"
        )

        # Nếu đang tìm kiếm thì báo số kết quả
        if search_title.strip():
            st.caption(
                f"Tìm thấy {len(movie_options)} phim phù hợp."
            )

        # CHỌN PHIM
        selected_movie = st.selectbox(
            "Chọn phim",
            movie_options["Title"].tolist(),
            key="movie_select"
        )

        # Lấy thông tin phim
        movie = movie_options[
            movie_options["Title"] == selected_movie
        ].iloc[0]

        st.divider()

        # TÊN + THÔNG TIN CƠ BẢN
        st.header(movie["Title"])

        st.write(
            f"**Đạo diễn:** {movie['Director']}"
        )

        st.write(
            f"**Thể loại:** {movie['Genre']}"
        )

        st.write(
            f"**Diễn viên:** {movie['Actors']}"
        )

        # CÁC CHỈ SỐ CHÍNH
        col1, col2, col3, col4 = st.columns(4)

        with col1:

            if pd.notna(movie["Year"]):
                st.metric(
                    "Năm phát hành",
                    int(movie["Year"])
                )
            else:
                st.metric(
                    "Năm phát hành",
                    "N/A"
                )

        with col2:

            if pd.notna(movie["Runtime"]):
                st.metric(
                    "Thời lượng",
                    f"{int(movie['Runtime'])} phút"
                )
            else:
                st.metric(
                    "Thời lượng",
                    "N/A"
                )

        with col3:

            if pd.notna(movie["Rating"]):
                st.metric(
                    "IMDb Rating",
                    f"{movie['Rating']:.1f}/10"
                )
            else:
                st.metric(
                    "IMDb Rating",
                    "N/A"
                )

        with col4:

            if pd.notna(movie["Metascore"]):
                st.metric(
                    "Metascore",
                    f"{movie['Metascore']:.0f}/100"
                )
            else:
                st.metric(
                    "Metascore",
                    "N/A"
                )

        # VOTES + REVENUE
        col1, col2 = st.columns(2)

        with col1:

            if pd.notna(movie["Votes"]):
                st.metric(
                    "Lượt bình chọn",
                    f"{movie['Votes']:,.0f}"
                )
            else:
                st.metric(
                    "Lượt bình chọn",
                    "N/A"
                )

        with col2:

            if pd.notna(movie["Revenue"]):
                st.metric(
                    "Doanh thu",
                    f"{movie['Revenue']:,.2f} triệu USD"
                )
            else:
                st.metric(
                    "Doanh thu",
                    "N/A"
                )

        # MÔ TẢ
        st.write("### Nội dung phim")

        if pd.notna(movie["Description"]):
            st.write(movie["Description"])
        else:
            st.write("Không có mô tả.")

with tab7:
    st.subheader("Dự đoán doanh thu phim")

    st.caption(
        "Nhập các thông tin của phim để ước tính doanh thu bằng mô hình Linear Regression."
    )

    st.write("### Hiệu suất mô hình")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "MAE",
            "50.68 triệu USD"
        )
    
    with col2:
        st.metric(
            "RMSE",
            "79.89 triệu USD"
        )
    
    with col3:
        st.metric(
            "R²",
            "0.5109"
        )
    
    st.caption(
        "Các chỉ số đánh giá được tính trên tập kiểm tra (20% dữ liệu), "
        "chỉ sử dụng các phim có thông tin doanh thu."
    )
    
    st.divider()
    
    st.write("### Nhập thông tin phim")

    
    # NHẬP THÔNG TIN PHIM
    with st.form("revenue_form"):
        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input(
                "Năm phát hành",
                min_value=int(df["Year"].min()),
                max_value=int(df["Year"].max()),
                value=int(df["Year"].median()),
                step=1
            )

            runtime = st.number_input(
                "Thời lượng phim (phút)",
                min_value=int(df["Runtime"].min()),
                max_value=int(df["Runtime"].max()),
                value=int(df["Runtime"].median()),
                step=1
            )

            rating = st.number_input(
                "IMDb Rating",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1
            )

        with col2:
            votes = st.number_input(
                "Số lượt bình chọn (Votes)",
                min_value=0,
                value=int(df["Votes"].median()),
                step=1000
            )

            metascore = st.number_input(
                "Metascore",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0
            )

        submitted = st.form_submit_button(
            "Dự đoán doanh thu",
            type="primary"
        )

    # DỰ ĐOÁN
    if submitted:
        input_data = pd.DataFrame([{
            "Year": year,
            "Runtime": runtime,
            "Rating": rating,
            "Votes": votes,
            "Metascore": metascore
        }])

        predicted_revenue = linear_model.predict(input_data)[0]

        st.divider()
        
        if predicted_revenue < 0:
            st.warning(
                "Mô hình dự đoán doanh thu âm. "
                "Kết quả này không có ý nghĩa thực tế, "
                "cho thấy hạn chế của Linear Regression."
            )
        else:
            st.metric(
                "Doanh thu dự đoán",
                f"{predicted_revenue:,.2f} triệu USD"
            )

with tab8:
    st.subheader("Dữ liệu phim IMDb")

    # THÔNG TIN TỔNG QUAN
    total_data = len(df)
    filtered_data = len(filtered_df)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Tổng số phim",
            f"{total_data:,}"
        )

    with col2:
        st.metric(
            "Số phim sau bộ lọc",
            f"{filtered_data:,}"
        )

    st.divider()

    # CHỌN CỘT HIỂN THỊ
    default_columns = [
        "Title",
        "Year",
        "Genre",
        "Director",
        "Runtime",
        "Rating",
        "Votes",
        "Revenue",
        "Metascore"
    ]

    selected_columns = st.multiselect(
        "Cột hiển thị",
        options=df.columns.tolist(),
        default=default_columns,
        key="data_columns"
    )

    # HIỂN THỊ BẢNG
    if not has_data:
        st.warning(
            "Không có dữ liệu phù hợp với bộ lọc hiện tại."
        )
    
    elif selected_columns:
        display_df = (
            filtered_df[selected_columns]
            .reset_index(drop=True)
        )
    
        display_df.index = display_df.index + 1
    
        st.dataframe(
            display_df,
            use_container_width=True,
            height=550
        )
    
    else:
        st.info(
            "Hãy chọn ít nhất một cột để hiển thị dữ liệu."
        )
