import tkinter as tk
from tkinter import filedialog, ttk
import ttkbootstrap as tb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Create main window
root = tb.Window(themename="darkly")
root.title("Teacher Dashboard - Attendance & Marks Visualizer")
root.geometry("1250x700")

# Global variables
df = None
selected_analysis = tk.StringVar(value="Select")
selected_column1 = tk.StringVar(value="Select")
selected_column2 = tk.StringVar(value="Select")

# --- Upload CSV File ---
def upload_file():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        status_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
        update_table()
        columns = list(df.columns)
        column_selector1['values'] = ["Select"] + columns
        column_selector2['values'] = ["Select"] + columns

# --- Update Table (Compact) ---
def update_table():
    for i in tree.get_children():
        tree.delete(i)
    if df is not None:
        tree["columns"] = list(df.columns)
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")
        # Show limited rows to save space
        for _, row in df.head(10).iterrows():
            tree.insert("", "end", values=list(row))
    else:
        tree["columns"] = []
        tree.insert("", "end", values=["No Data Loaded"])

# --- Generate Visualization ---
def generate_analysis():
    if df is None:
        status_label.config(text="Please upload a CSV file first!")
        return

    analysis_type = selected_analysis.get()
    col1, col2 = selected_column1.get(), selected_column2.get()

    if analysis_type == "Select":
        status_label.config(text="Please select an analysis type.")
        return

    plt.clf()

    try:
        if analysis_type == "Frequency Analysis" and col1 != "Select":
            freq = df[col1].value_counts().head(10)
            sns.barplot(x=freq.index, y=freq.values, palette="crest")
            plt.title(f"Top 10 Frequency of {col1}")
            plt.xticks(rotation=45)

        elif analysis_type == "Correlation Matrix":
            corr = df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Matrix of Numerical Columns")

        elif analysis_type == "Marks Summary":
            summary = df.describe(numeric_only=True).T[['mean', 'std', 'min', 'max']]
            summary.plot(kind="bar", figsize=(8, 5), legend=True)
            plt.title("Marks Summary (Mean, Std, Min, Max)")

        elif analysis_type == "Custom Comparison" and col1 != "Select" and col2 != "Select":
            sns.scatterplot(x=df[col1], y=df[col2], s=60, color="#0dcaf0", edgecolor="white")
            plt.title(f"{col1} vs {col2} Comparison")
            plt.xlabel(col1)
            plt.ylabel(col2)

        elif analysis_type == "Trend Analysis" and col1 != "Select":
            df[col1].value_counts().sort_index().plot(kind="line", marker='o')
            plt.title(f"Trend of {col1} Over Time")
            plt.xlabel("Index")
            plt.ylabel("Count")

        else:
            status_label.config(text="Please select valid columns for analysis.")
            return

        display_plot()

    except Exception as e:
        status_label.config(text=f"Error: {e}")

# --- Display Plot in Tkinter Frame ---
def display_plot():
    for widget in graph_frame.winfo_children():
        widget.destroy()
    fig = plt.gcf()
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

# --- UI Components ---
title_label = tb.Label(root, text="📘 Teacher Dashboard", font=("Helvetica", 18, "bold"))
title_label.pack(pady=10)

upload_btn = tb.Button(root, text="Upload CSV File", bootstyle="success", command=upload_file)
upload_btn.pack(pady=5)

# --- Main Frame Layout ---
main_frame = tb.Frame(root)
main_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Left Control Panel
control_frame = tb.LabelFrame(main_frame, text="⚙️ Controls", padding=10)
control_frame.pack(side='left', fill='y', padx=10, pady=5)

ttk.Label(control_frame, text="Select Analysis Type:").pack(pady=5)
analysis_menu = ttk.Combobox(control_frame, textvariable=selected_analysis, values=[
    "Select", "Frequency Analysis", "Correlation Matrix", "Marks Summary", "Custom Comparison", "Trend Analysis"
])
analysis_menu.pack(pady=5)

ttk.Label(control_frame, text="Select Column 1:").pack(pady=5)
column_selector1 = ttk.Combobox(control_frame, textvariable=selected_column1)
column_selector1.pack(pady=5)

ttk.Label(control_frame, text="Select Column 2 (optional):").pack(pady=5)
column_selector2 = ttk.Combobox(control_frame, textvariable=selected_column2)
column_selector2.pack(pady=5)

generate_btn = tb.Button(control_frame, text="Generate Graph", bootstyle="info", command=generate_analysis)
generate_btn.pack(pady=10)

status_label = tb.Label(control_frame, text="No file loaded", bootstyle="secondary")
status_label.pack(pady=10)

# Data Preview (Compact)
table_frame = tb.LabelFrame(main_frame, text="📋 Data Preview (Top 10 Rows)", padding=10)
table_frame.pack(side='left', fill='both', expand=True, padx=10)

tree = ttk.Treeview(table_frame, show='headings')
tree.pack(fill='both', expand=True)

# Graph Display
graph_frame = tb.LabelFrame(root, text="📈 Data Visualization", padding=10)
graph_frame.pack(fill='both', expand=True, padx=10, pady=10)

tb.Label(graph_frame, text="Graphs will appear here", bootstyle="secondary").pack(pady=20)

root.mainloop()
