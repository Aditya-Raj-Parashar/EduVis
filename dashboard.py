import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class TeacherDashboard(tb.Window):
    def __init__(self):
        super().__init__(themename="superhero")  # cyborg, darkly, solar, morph, journal

        self.title("Teacher Dashboard - Attendance & Marks Visualizer")
        self.geometry("1000x650")
        self.resizable(True, True)

        # Initialize dataframe
        self.df = None

        # Create UI components
        self.create_menu()
        self.create_widgets()
        self.create_graph_placeholder()

        # ------------------- VISUALIZATION FUNCTIONS -------------------
    def clear_canvas(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

    def update_secondary_options(self, event=None):
        """Update column selection dropdowns when visualization type changes"""
        if self.df is None:
            messagebox.showinfo("No Data", "Please upload a CSV file first!")
            return

        numeric_cols = list(self.df.select_dtypes(include='number').columns)
        if self.visual_type.get() == "Correlation":
            self.col1_menu.config(values=numeric_cols)
            self.col2_menu.config(values=numeric_cols, state="readonly")
        else:
            self.col1_menu.config(values=numeric_cols)
            self.col2_menu.set("N/A")
            self.col2_menu.config(state="disabled")

    def generate_graph(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please upload a CSV file first.")
            return

        vis_type = self.visual_type.get()
        self.clear_canvas()

        fig = Figure(figsize=(6, 3), dpi=100)
        ax = fig.add_subplot(111)

        try:
            if vis_type == "Marks Distribution":
                numeric_cols = self.df.select_dtypes(include='number').columns
                if len(numeric_cols) > 0:
                    self.df[numeric_cols].mean().plot(kind='bar', ax=ax, color='orange')
                    ax.set_title("Average Marks per Subject")
                    ax.set_ylabel("Average Marks")
                else:
                    ax.text(0.5, 0.5, "No numeric data found", ha="center", va="center")

            elif vis_type == "Attendance Trend":
                numeric_cols = self.df.select_dtypes(include='number').columns
                if len(numeric_cols) > 1:
                    self.df[numeric_cols].mean().plot(kind='line', marker='o', ax=ax)
                    ax.set_title("Attendance Trend Over Time")
                    ax.set_ylabel("Average Attendance")
                else:
                    ax.text(0.5, 0.5, "Not enough numeric columns", ha="center", va="center")

            elif vis_type == "Correlation":
                col1 = self.col1.get()
                col2 = self.col2.get()

                if col1 in self.df.columns and col2 in self.df.columns:
                    ax.scatter(self.df[col1], self.df[col2], color='cyan')
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    ax.set_title(f"Correlation between {col1} and {col2}")
                else:
                    ax.text(0.5, 0.5, "Select two valid numeric columns", ha="center", va="center")

            else:
                ax.text(0.5, 0.5, "Please select a visualization type", ha="center", va="center")

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def clear_canvas(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

    def show_marks_distribution(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please upload a CSV first.")
            return

        self.clear_canvas()
        fig = Figure(figsize=(6, 3), dpi=100)
        ax = fig.add_subplot(111)

        # pick numeric columns only (like marks)
        numeric_cols = self.df.select_dtypes(include='number').columns
        if len(numeric_cols) == 0:
            ax.text(0.5, 0.5, "No numeric data found", ha="center", va="center")
        else:
            self.df[numeric_cols].mean().plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title("Average Marks per Subject")
            ax.set_ylabel("Average Marks")

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def show_attendance_trend(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please upload a CSV first.")
            return

        self.clear_canvas()
        fig = Figure(figsize=(6, 3), dpi=100)
        ax = fig.add_subplot(111)

        # Assuming months columns are attendance numbers
        numeric_cols = self.df.select_dtypes(include='number').columns
        if len(numeric_cols) > 1:
            self.df[numeric_cols].mean().plot(kind='line', marker='o', ax=ax)
            ax.set_title("Attendance Trend Over Time")
            ax.set_xlabel("Months")
            ax.set_ylabel("Average Attendance")
        else:
            ax.text(0.5, 0.5, "No attendance-like data found", ha="center", va="center")

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def show_correlation(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please upload a CSV first.")
            return

        self.clear_canvas()
        fig = Figure(figsize=(6, 3), dpi=100)
        ax = fig.add_subplot(111)

        numeric_cols = self.df.select_dtypes(include='number').columns
        if len(numeric_cols) > 1:
            corr = self.df[numeric_cols].corr()
            cax = ax.matshow(corr, cmap="coolwarm")
            fig.colorbar(cax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns)
            ax.set_title("Correlation Heatmap", pad=15)
        else:
            ax.text(0.5, 0.5, "Not enough numeric data", ha="center", va="center")

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)


    # ------------------- MENU BAR -------------------
    def create_menu(self):
        menubar = tb.Menu(self)
        file_menu = tb.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open CSV", command=self.load_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    # ------------------- MAIN LAYOUT -------------------
    def create_widgets(self):
        # Title
        title_label = tb.Label(
            self, text="📊 Teacher Dashboard", font=("Segoe UI", 20, "bold"), bootstyle="inverse-dark"
        )
        title_label.pack(pady=10)

        # Upload Button
        upload_btn = tb.Button(
            self,
            text="Upload CSV File",
            bootstyle="success-outline",
            command=self.load_csv,
            width=20
        )
        upload_btn.pack(pady=5)

        # Table Frame
        self.table_frame = tb.Frame(self)
        self.table_frame.pack(pady=10, fill=BOTH, expand=True)

    # ------------------- LOAD CSV -------------------
    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path)
            self.display_table(self.df)
            messagebox.showinfo("Success", "CSV file loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")

    # ------------------- DISPLAY DATA IN TREEVIEW -------------------
    def display_table(self, df):
        # Clear previous table
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        # Create Treeview
        cols = list(df.columns)
        tree = tb.Treeview(self.table_frame, columns=cols, show="headings", bootstyle="dark")

        # Define columns
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor=CENTER)

        # Insert rows
        for _, row in df.iterrows():
            tree.insert("", END, values=list(row))

        # Add scrollbar
        scrollbar = tb.Scrollbar(self.table_frame, orient=VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill=Y)
        tree.pack(fill=BOTH, expand=True)

    # ------------------- ADVANCED GRAPH SECTION -------------------
    def create_graph_placeholder(self):
        self.graph_frame = tb.Labelframe(
            self,
            text="📈 Data Visualization",
            bootstyle="info",
            padding=10
        )
        self.graph_frame.pack(pady=10, fill=BOTH, expand=True)

        # Control Panel Frame
        control_frame = tb.Frame(self.graph_frame)
        control_frame.pack(pady=5, fill=X)

        # Visualization type dropdown
        tb.Label(control_frame, text="Select Visualization:", bootstyle="inverse-info").pack(side=LEFT, padx=5)
        self.visual_type = tb.StringVar(value="Select Option")
        vis_options = ["Marks Distribution", "Attendance Trend", "Correlation"]
        vis_menu = tb.Combobox(control_frame, textvariable=self.visual_type, values=vis_options, state="readonly", width=20)
        vis_menu.pack(side=LEFT, padx=5)
        vis_menu.bind("<<ComboboxSelected>>", self.update_secondary_options)

        # Secondary selection (columns)
        tb.Label(control_frame, text="Select Columns:", bootstyle="inverse-info").pack(side=LEFT, padx=5)
        self.col1 = tb.StringVar(value="Select Column 1")
        self.col2 = tb.StringVar(value="Select Column 2")

        self.col1_menu = tb.Combobox(control_frame, textvariable=self.col1, values=[], state="readonly", width=20)
        self.col1_menu.pack(side=LEFT, padx=5)
        self.col2_menu = tb.Combobox(control_frame, textvariable=self.col2, values=[], state="readonly", width=20)
        self.col2_menu.pack(side=LEFT, padx=5)

        # Generate Button
        tb.Button(control_frame, text="Generate Graph", bootstyle="success-outline", command=self.generate_graph).pack(side=LEFT, padx=10)

        # Canvas area
        self.canvas_frame = tb.Frame(self.graph_frame)
        self.canvas_frame.pack(fill=BOTH, expand=True)




# ------------------- RUN APP -------------------
if __name__ == "__main__":
    app = TeacherDashboard()
    app.mainloop()
