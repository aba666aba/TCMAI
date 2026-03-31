import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
import json
import threading
from tcm_complete_gnn_model_fixed import TCMGNNModel
from tcm_gnn_training_complete import TCMGraphDataBuilder

class TCMModelInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("中药处方推荐系统")
        self.root.geometry("800x600")
        
        # 模型和数据相关变量
        self.model = None
        self.graph_data = None
        self.clean_data = None
        self.herb_list = []
        self.symptom_to_id = {}
        
        # 创建界面
        self.create_widgets()
        
        # 初始化模型
        self.init_model()
    
    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="中药处方推荐系统", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 症状输入框架
        input_frame = ttk.LabelFrame(main_frame, text="症状描述", padding="10")
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # 症状输入
        self.symptom_entry = ttk.Entry(input_frame, width=50, font=("Arial", 12))
        self.symptom_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.symptom_entry.bind("<Return>", self.predict_prescription)
        
        # 预测按钮
        predict_button = ttk.Button(input_frame, text="推荐处方", command=self.predict_prescription)
        predict_button.grid(row=0, column=1)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(main_frame, text="推荐结果", padding="10")
        result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # 结果显示区域
        self.result_text = scrolledtext.ScrolledText(result_frame, width=80, height=20, font=("Arial", 11))
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def init_model(self):
        """初始化模型"""
        try:
            self.status_var.set("正在加载模型...")
            self.root.update()
            
            # 在后台线程中加载模型
            threading.Thread(target=self.load_model_async, daemon=True).start()
        except Exception as e:
            messagebox.showerror("错误", f"初始化模型时出错: {str(e)}")
            self.status_var.set("模型初始化失败")
    
    def load_model_async(self):
        """在后台线程中加载模型"""
        try:
            # 加载清洗后的数据
            with open("tcm_cleaned_data.json", 'r', encoding='utf-8') as f:
                self.clean_data = json.load(f)
            
            # 构建图数据
            graph_builder = TCMGraphDataBuilder(self.clean_data)
            self.graph_data = graph_builder.build_graph_data()
            
            # 创建模型
            entity_type_sizes = {
                'Herb': len(self.clean_data['herbs']),
                'Symptom': len(self.clean_data['symptoms']),
                'Disease': len(self.clean_data['diseases']),
                'Effect': len(self.clean_data['effects'])
            }
            
            self.model = TCMGNNModel(
                entity_type_sizes=entity_type_sizes,
                relation_type_size=len(graph_builder.relation_types),
                embedding_dim=128,
                gnn_hidden_dim=256,
                gnn_output_dim=128,
                herb_vocab_size=len(self.clean_data['herbs']),
                symptom_vocab_size=len(self.clean_data['symptoms']),
                max_prescription_length=15,
                num_gnn_layers=3,
                gnn_type='sage',
                dropout=0.1
            )
            
            # 加载训练好的模型权重
            checkpoint = torch.load("tcm_gnn_model.pth", map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # 设置相关变量
            self.herb_list = [herb['name'] for herb in self.clean_data['herbs']]
            self.symptom_to_id = {symptom['name']: idx for idx, symptom in enumerate(self.clean_data['symptoms'])}
            
            # 更新UI
            self.root.after(0, self.on_model_loaded)
        except Exception as e:
            self.root.after(0, lambda: self.on_model_load_error(str(e)))
    
    def on_model_loaded(self):
        """模型加载完成后的回调"""
        self.status_var.set("模型加载完成，可以开始使用")
        self.result_text.insert(tk.END, "模型加载成功！请输入症状描述以获取处方推荐。\n\n")
        self.result_text.insert(tk.END, "示例症状：头痛、发热、咳嗽、腹痛等\n")
        self.result_text.see(tk.END)
    
    def on_model_load_error(self, error_msg):
        """模型加载失败的回调"""
        messagebox.showerror("错误", f"加载模型时出错: {error_msg}")
        self.status_var.set("模型加载失败")
        self.result_text.insert(tk.END, f"模型加载失败: {error_msg}\n")
        self.result_text.see(tk.END)
    
    def predict_prescription(self, event=None):
        """预测处方"""
        if self.model is None:
            messagebox.showwarning("警告", "模型尚未加载完成，请稍后再试")
            return
        
        # 获取症状输入
        symptom_text = self.symptom_entry.get().strip()
        if not symptom_text:
            messagebox.showwarning("警告", "请输入症状描述")
            return
        
        # 在后台线程中进行预测
        self.status_var.set("正在生成处方推荐...")
        self.root.update()
        
        threading.Thread(target=self.predict_async, args=(symptom_text,), daemon=True).start()
    
    def predict_async(self, symptom_text):
        """在后台线程中进行预测"""
        try:
            # 将症状文本转换为ID
            symptom_id = self.symptom_to_id.get(symptom_text, 0)
            symptom_tokens = torch.tensor([[symptom_id]], dtype=torch.long)
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(
                    symptom_tokens=symptom_tokens,
                    kg_entity_ids=self.graph_data.node_ids,
                    kg_entity_types=self.graph_data.node_types,
                    kg_edge_index=self.graph_data.edge_index
                )
            
            # 获取处方得分
            prescription_scores = outputs['prescription_scores']
            
            # 根据得分排序，获取推荐的药材
            top_indices = torch.argsort(prescription_scores[0], descending=True)[:10]  # 获取前10个推荐
            
            # 将ID转换为药材名称
            recommended_herbs = [self.herb_list[idx.item()] for idx in top_indices]
            scores = [prescription_scores[0][idx].item() for idx in top_indices]
            
            # 更新UI
            self.root.after(0, lambda: self.display_results(symptom_text, recommended_herbs, scores))
        except Exception as e:
            self.root.after(0, lambda: self.display_error(str(e)))
    
    def display_results(self, symptom_text, herbs, scores):
        """显示推荐结果"""
        self.status_var.set("处方推荐完成")
        
        # 清空之前的结果
        self.result_text.delete(1.0, tk.END)
        
        # 显示结果
        self.result_text.insert(tk.END, f"症状描述: {symptom_text}\n\n")
        self.result_text.insert(tk.END, "推荐处方:\n")
        self.result_text.insert(tk.END, "-" * 50 + "\n")
        
        for i, (herb, score) in enumerate(zip(herbs, scores), 1):
            self.result_text.insert(tk.END, f"{i:2d}. {herb:<20} (得分: {score:.4f})\n")
        
        self.result_text.insert(tk.END, "\n" + "-" * 50 + "\n")
        self.result_text.insert(tk.END, "注意：以上推荐仅供参考，实际用药请咨询专业中医师。\n")
        self.result_text.see(tk.END)
    
    def display_error(self, error_msg):
        """显示错误信息"""
        self.status_var.set("处方推荐失败")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"处方推荐失败: {error_msg}\n")
        self.result_text.see(tk.END)

def main():
    root = tk.Tk()
    app = TCMModelInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()