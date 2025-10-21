import tkinter as tk
from tkinter import filedialog, ttk
import ttkbootstrap as tb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Create window
# cyborg, darkly, solar, morph, journal
root = tb.Window(themename="darkly")
root.title("Teacher Dashboard - Attendance & Marks Visualizer")
root.geometry("1200x700")

# Variables
df = None
selected_analysis = tk.StringVar()
selected_column1 = tk.StringVar()
selected_column2 = tk.StringVar()

# --- Upload Function ---
def upload_file():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        update_table()
        column_selector1['values'] = list(df.columns)
        column_selector2['values'] = list(df.columns)
        status_label.config(text=f"Loaded: {file_path.split('/')[-1]}")

# --- Update Table ---
def update_table():
    for i in tree.get_children():
        tree.delete(i)
    for index, row in df.head(20).iterrows():
        tree.insert("", "end", values=list(row))

# --- Generate Visualization ---
def generate_analysis():
    if df is None:
        status_label.config(text="Please upload a CSV file first!")
        return

    plt.clf()
    analysis_type = selected_analysis.get()

    if analysis_type == "Frequency Analysis":
        col = selected_column1.get()
        if col:
            freq = df[col].value_counts().head(10)
            sns.barplot(x=freq.index, y=freq.values)
            plt.title(f"Frequency of {col}")
            plt.xticks(rotation=45)

    elif analysis_type == "Correlation Matrix":
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")

    elif analysis_type == "Marks Summary":
        df.describe().T[['mean', 'std', 'min', 'max']].plot(kind='bar')
        plt.title("Marks Summary")

    elif analysis_type == "Custom Comparison":
        col1, col2 = selected_column1.get(), selected_column2.get()
        if col1 and col2:
            sns.scatterplot(x=df[col1], y=df[col2])
            plt.title(f"Scatter Plot: {col1} vs {col2}")

    display_plot()

# --- Display Plot in Canvas ---
def display_plot():
    for widget in graph_frame.winfo_children():
        widget.destroy()
    fig = plt.gcf()
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

# --- UI Layout ---
title_label = tb.Label(root, text="📘 Teacher Dashboard", font=("Helvetica", 18, "bold"))
title_label.pack(pady=10)

upload_btn = tb.Button(root, text="Upload CSV File", bootstyle="success", command=upload_file)
upload_btn.pack(pady=5)

main_frame = tb.Frame(root)
main_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Left Panel - Controls
control_frame = tb.LabelFrame(main_frame, text="⚙️ Controls")
control_frame.pack(side='left', fill='y', padx=5, pady=5)

ttk.Label(control_frame, text="Select Analysis:").pack(pady=5)
analysis_menu = ttk.Combobox(control_frame, textvariable=selected_analysis, values=[
    "Frequency Analysis", "Correlation Matrix", "Marks Summary", "Custom Comparison"
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

# Right Panel - Table
table_frame = tb.LabelFrame(main_frame, text="📋 Data Preview")
table_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)

columns = ("A", "B", "C", "D", "E")
tree = ttk.Treeview(table_frame, columns=columns, show='headings')
tree.pack(fill='both', expand=True)

# Bottom Frame - Graph Display
graph_frame = tb.LabelFrame(root, text="📈 Data Visualization")
graph_frame.pack(fill='both', expand=True, padx=10, pady=10)

tb.Label(graph_frame, text="Graphs will appear here", bootstyle="secondary").pack(pady=20)

root.mainloop()
