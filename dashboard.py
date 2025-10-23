import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import ttkbootstrap as tb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# --- Export Graph as Image ---
def export_graph():
    try:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
        if file_path:
            plt.savefig(file_path)
            messagebox.showinfo("Export Success", f"Graph saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Export Error", str(e))

# --- Filter/Search Data ---
def filter_table():
    query = filter_entry.get().lower()
    for i in tree.get_children():
        tree.delete(i)
    if df is not None and query != '':
        filtered = df[df.apply(lambda row: row.astype(str).str.lower().str.contains(query).any(), axis=1)]
        for _, row in filtered.head(10).iterrows():
            tree.insert("", "end", values=list(row))

# --- Missing Data Visualization ---
def plot_missing_data():
    if df is not None:
        plt.clf()
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing.plot(kind='bar', color='orange')
            plt.title('Missing Values per Column')
            plt.ylabel('Count')
            plt.tight_layout()
            display_plot()
        else:
            status_label.config(text="No missing data to display.")

# --- Download Insights ---
def export_insights():
    text = insights_text.get(1.0, tk.END)
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text File", "*.txt")])
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        messagebox.showinfo("Export Success", f"Insights saved to {file_path}")



def drop_duplicates():
    global df
    if df is not None:
        before = len(df)
        df = df.drop_duplicates()
        df = df.dropna()
        after = len(df)
        update_table()
        generate_insights()
        messagebox.showinfo(
            "Rows Removed",
            f"All duplicate and null rows have been dropped.\nRows before: {before}\nRows after: {after}"
        )


# --- Window setup ---
root = tb.Window(themename="darkly")
root.title("Teacher Dashboard - Attendance & Marks Visualizer")
root.geometry("1400x750")

# Variables
df = None
selected_analysis = tk.StringVar(value="Select")
selected_column1 = tk.StringVar(value="Select")
selected_column2 = tk.StringVar(value="Select")

# --- Update Table (compact preview) ---
def update_table():
    for i in tree.get_children():
        tree.delete(i)
    if df is not None:
        tree["columns"] = list(df.columns)
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))

def show_insights_popup():
    popup = tk.Toplevel(root)
    popup.title("💡 Smart Insights")
    popup.geometry("500x600")
    popup.transient(root)
    popup.grab_set()
    text = tk.Text(popup, height=30, wrap='word', bg="#2a2a2a", fg="white", relief="flat")
    text.pack(fill='both', expand=True, padx=10, pady=10)
    # Fill with current insights
    text.insert(tk.END, insights_text.get(1.0, tk.END))
    text.config(state='disabled')
    btn = tb.Button(popup, text="Close", bootstyle="danger", command=popup.destroy)
    btn.pack(pady=10)

# --- Upload Function ---
def upload_file():
    global df
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("CSV Files", "*.csv"),
            ("Excel Files", "*.xlsx;*.xls"),
            ("JSON Files", "*.json"),
            ("All Files", "*.*")
        ]
    )
    if file_path:
        try:
            if file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                df = pd.read_csv(file_path)
            status_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
            selected_analysis.set("Select")  # Reset analysis type
            selected_column1.set("Select")   # Reset columns
            selected_column2.set("Select")
            update_table()
            update_column_selectors()
            generate_insights()
            # Hide the insights_frame and show popup
            insights_frame.pack_forget()
            show_insights_popup()
        except Exception as e:
            messagebox.showerror("File Load Error", f"Could not load file:\n{e}")

# --- Update Dropdowns ---
def update_column_selectors():
    if df is not None:
        columns = ["Select"] + list(df.columns)
        column_selector1["values"] = columns
        column_selector2["values"] = columns

# --- Generate Insights ---
def generate_insights():
    if df is None:
        return

    insights_text.delete(1.0, tk.END)
    insights_text.insert(tk.END, "📊 Here are some smart insights from your data! <3\n\n")

    try:
        # General Info
        insights_text.insert(tk.END, f"Your dataset has {len(df)} records and {len(df.columns)} columns. o.o\n\n")

        # Data types
        dtypes = df.dtypes.astype(str)
        insights_text.insert(tk.END, f"Let's look at the data types for each column: 8-]\n")
        for col, dtype in dtypes.items():
            insights_text.insert(tk.END, f"- {col}: {dtype}\n")
        insights_text.insert(tk.END, "\n")

        # Columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            insights_text.insert(tk.END, f"Some columns have missing values: {', '.join(missing_cols)}. Please check below! :o\n")
            for col in missing_cols:
                count = df[col].isnull().sum()
                insights_text.insert(tk.END, f"  - {col}: {count} missing values\n")
        else:
            insights_text.insert(tk.END, "Awesome! No columns have missing values. ^_^\n\n")

        # Columns with duplicates
        duplicate_cols = []
        for col in df.columns:
            if df[col].duplicated().any():
                duplicate_cols.append(col)
        if duplicate_cols:
            insights_text.insert(tk.END, f"Heads up! These columns have duplicate values: {', '.join(duplicate_cols)}. 0_0\n")
            for col in duplicate_cols:
                count = df[col].duplicated().sum()
                insights_text.insert(tk.END, f"  - {col}: {count} duplicates\n")
        else:
            insights_text.insert(tk.END, "No duplicate values found in any column. Nice! :)\n\n")

        # Detect numeric columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            insights_text.insert(tk.END, f"Numeric columns detected: {', '.join(numeric_cols)}. Let's crunch some numbers! 8-]\n\n")

            # Basic stats
            summary = df[numeric_cols].describe().T
            top_mark_col = summary['mean'].idxmax()
            low_mark_col = summary['mean'].idxmin()

            insights_text.insert(tk.END, f"Column with the highest average: {top_mark_col} ({summary.loc[top_mark_col, 'mean']:.2f}) <3\n")
            insights_text.insert(tk.END, f"Column with the lowest average: {low_mark_col} ({summary.loc[low_mark_col, 'mean']:.2f}) o.o\n\n")

            # Missing data
            missing = df.isnull().sum().sum()
            if missing > 0:
                insights_text.insert(tk.END, f"There are {missing} missing values in total. Please review! :o\n\n")

            # Correlation check
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr().abs()
                strongest = corr.unstack().sort_values(ascending=False)
                first_pair = strongest[strongest < 1].idxmax()
                corr_val = corr.loc[first_pair[0], first_pair[1]]
                insights_text.insert(tk.END, f"The strongest correlation is between {first_pair[0]} and {first_pair[1]} ({corr_val:.2f}). Interesting! ^_^\n\n")

        # Attendance specific
        attendance_cols = [col for col in df.columns if "attend" in col.lower()]
        if attendance_cols:
            col = attendance_cols[0]
            avg_att = df[col].mean()
            insights_text.insert(tk.END, f"Average attendance in '{col}' is {avg_att:.2f}%. Keep it up! :D\n\n")

        # Marks specific
        marks_cols = [col for col in df.columns if "mark" in col.lower() or "score" in col.lower()]
        if marks_cols:
            col = marks_cols[0]
            top_student = df.loc[df[col].idxmax(), df.columns[0]]
            insights_text.insert(tk.END, f"The top scorer in '{col}' is {top_student} with a score of {df[col].max()}! 8-]\n\n")

    except Exception as e:
        insights_text.insert(tk.END, f"Dekh ab kya gad bad kar di : {e} :(")

# ----- Generate Graphs ----
def generate_analysis():
    if df is None:
        status_label.config(text="Yaar file to daal de 😭!")
        return

    analysis_type = selected_analysis.get()
    col1, col2 = selected_column1.get(), selected_column2.get()
    plt.clf()

    if analysis_type == "Select":
        status_label.config(text="Bhoot ka analysis karega? ")
        return

    try:
        if analysis_type == "Frequency Analysis" and col1 != "Select":
            freq = df[col1].value_counts().head(10)
            sns.barplot(x=freq.index, y=freq.values, hue=freq.index, palette="crest", legend=False)
            plt.title(f"Top 10 Frequency of {col1}")
            plt.xticks(rotation=45)

        elif analysis_type == "Correlation Matrix":
            corr = df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Matrix")

        elif analysis_type == "Marks Summary":
            summary = df.describe().T[['mean', 'std', 'min', 'max']]
            summary.plot(kind="bar", figsize=(8, 5), legend=True)
            plt.title("Marks Summary")

        elif analysis_type == "Custom Comparison" and col1 != "Select" and col2 != "Select":
            sns.scatterplot(x=df[col1], y=df[col2], s=60, color="#0dcaf0", edgecolor="white")
            plt.title(f"{col1} vs {col2}")
            plt.xlabel(col1)
            plt.ylabel(col2)

        elif analysis_type == "Trend Analysis" and col1 != "Select":
            df[col1].value_counts().sort_index().plot(kind="line", marker='o')
            plt.title(f"Trend of {col1}")

        elif analysis_type == "distribution" and col1 != "Select" :
            sns.histplot(df[col1].dropna(),kde = True, color="#1ad5fa", edgecolor="white")
            plt.title(f"Distribution of {col1}")
            plt.xlabel(col1)
            plt.ylabel("Frequency")

        elif analysis_type == "boxplot - outliers" and col1 != "Select" :
            sns.boxplot(x=df[col1].dropna(), color="#12e343", edgecolor="black")
            plt.title(f"boxplot of {col1}")
            plt.xlabel(col1)

        elif analysis_type == "Top N Categories" and col1 != "Select":
            n=10 #
            try:
                n_str = tk.simpledialog.askstring("Top N", "Enter the value of N (number of top categories to show):", parent=root)
                if n_str is not None and n_str.isdigit() and int(n_str) > 0:
                    n = int(n_str)
            except Exception as e:
                messagebox.showwarning("Input Error", f"Aukaat ke bahar ka N mat daaal warna.\n hello papa yaar system ki to ma.... \n Ab default N=10 use kar rha hu\n ye tera error : {e}")
            top_n = df[col1].value_counts().head(n)
            sns.barplot(x=top_n.values, y=top_n.index, palette="crest")
            plt.title(f"Top {n} Categories in {col1}")
            plt.xlabel("Count")
            plt.ylabel(col1)

        elif analysis_type == "Pairplot":
            sns.pairplot(df.select_dtypes(include="number"))
            plt.suptitle("Pairplot of Numeric Columns", y=1.02, x=1.02)

        elif analysis_type == "Summary Statistics":
            stats = df.describe().T
            insights_text.delete(1.0, tk.END)
            insights_text.insert(tk.END, stats.to_string())
            plt.clf()
            plt.axis('off')

        else:
            status_label.config(text="Mamma will be not so proud of you!!!! 😔")
            return
        plt.tight_layout()
        display_plot()
        generate_insights()

    except Exception as e:
        status_label.config(text=f"Error: {e}")
        messagebox.showerror("Dekh orr samajh kaha haga h. Maybe dataset me issue ho", str(e))

# ---- Display Plot -----
def display_plot():
    for widget in graph_frame.winfo_children():
        widget.destroy()
    fig = plt.gcf()
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

# --- Responsive Layout ---
# def on_resize(event=None):
#     width = root.winfo_width()
#     height = root.winfo_height()
#     # Adjust column widths in table preview
#     if df is not None and len(df.columns) > 0:
#         col_width = max(80, int((table_frame.winfo_width() - 40) / len(df.columns)))
#         for col in df.columns:
#             tree.column(col, width=col_width)

# root.bind('<Configure>', on_resize)

# gaand maraye Responsiveness
# mkb mujhe nhi karna yeeeeee....




# --- UI Components ---
title_label = tb.Label(root, text="📘 Teacher Dashboard", font=("Helvetica", 18, "bold"))
title_label.pack(pady=10)

upload_btn = tb.Button(root, text="Upload CSV File", bootstyle="success", command=upload_file)
upload_btn.pack(pady=5)

# --- All buttons and Fucking peice of SHITT.......... ---
main_frame = tb.Frame(root)
main_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Left Controls
control_frame = tb.LabelFrame(main_frame, text="⚙️ Controls", padding=10)
control_frame.pack(side='left', fill='y', padx=10, pady=5)

ttk.Label(control_frame, text="Select Analysis Type:").pack(pady=5)
analysis_menu = ttk.Combobox(control_frame, textvariable=selected_analysis, values=[
    "Select", "Frequency Analysis", "Correlation Matrix", "Marks Summary", "Custom Comparison", "Trend Analysis",
    "distribution", "boxplot - outliers", "Top N Categories", "Pairplot", "Summary Statistics"
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

export_btn = tb.Button(control_frame, text="Export Graph", bootstyle="warning", command=export_graph)
export_btn.pack(pady=5)

filter_label = ttk.Label(control_frame, text="Filter/Search:")
filter_label.pack(pady=5)
filter_entry = ttk.Entry(control_frame)
filter_entry.pack(pady=5)
filter_btn = tb.Button(control_frame, text="Apply Filter", bootstyle="primary", command=filter_table)
filter_btn.pack(pady=5)

missing_data_btn = tb.Button(control_frame, text="Plot Missing Data", bootstyle="danger", command=plot_missing_data)
missing_data_btn.pack(pady=5)

drop_duplicates_btn = tb.Button(control_frame, text="Drop Duplicates & Nulls", bootstyle="danger", command=drop_duplicates)
drop_duplicates_btn.pack(pady=5)
export_insights_btn = tb.Button(control_frame, text="Download Insights", bootstyle="success", command=export_insights)
export_insights_btn.pack(pady=5)

status_label = tb.Label(control_frame, text="No file loaded", bootstyle="secondary")
status_label.pack(pady=10)

# Graph Section
graph_frame = tb.LabelFrame(main_frame, text="📈 Data Visualization", padding=20)
graph_frame.pack(fill='both', expand=True, padx=10, pady=10)

tb.Label(graph_frame, text="Graphs will appear here", bootstyle="secondary").pack(pady=20)

# Center Table
table_frame = tb.LabelFrame(main_frame, text="Data Preview ", padding=10)
table_frame.pack(side='left', fill='both' , expand=True, padx=10)

# Add a horizontal scrollbar
tree_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal")
tree_scroll_x.pack(side='bottom', fill='x')

# Add a vertical scrollbar
tree_scroll_y = ttk.Scrollbar(table_frame, orient="vertical")
tree_scroll_y.pack(side='right', fill='y')

tree = ttk.Treeview(
    table_frame,
    show='headings',
    xscrollcommand=tree_scroll_x.set,
    yscrollcommand=tree_scroll_y.set
)
tree.pack(fill='both', expand=True)
tree_scroll_x.config(command=tree.xview)
tree_scroll_y.config(command=tree.yview)

# Right Smart Insights
insights_frame = tb.LabelFrame(main_frame, text="💡 Smart Insights", padding=10)
insights_frame.pack(side='right', fill='both', expand=True, padx=10, pady=5)

insights_text = tk.Text(insights_frame, height=25, wrap='word', bg="#2a2a2a", fg="white", relief="flat")
insights_text.pack(fill='both', expand=True)



def on_analysis_change(event=None):
    analysis = selected_analysis.get()
    if analysis in ["Correlation Matrix", "Marks Summary",'pairplot']:
        column_selector1.config(state="disabled")
        column_selector2.config(state="disabled")
    elif analysis in ["Custom Comparison","Trend Analysis","distribution", "Top N Categories", "Summary Statistics"]:
        column_selector1.config(state="readonly")
        column_selector2.config(state="readonly")
    else:
        column_selector1.config(state="readonly")
        column_selector2.config(state="disabled")
# "Custom Comparison", "Trend Analysis",
    #"distribution", "boxplot - outliers", "Top N Categories", "Pairplot", "Summary Statistics"
analysis_menu.bind("<<ComboboxSelected>>", on_analysis_change)

root.mainloop()






#Faaaaaaaaaaaaaakkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
#iske baad mai tkinter ka project kabhi nhi karunga