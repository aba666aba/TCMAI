"""测试启动画面"""
import sys, os
ROOT = r'E:\Tom\TCMAI'
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.environ['TCMAI_ROOT'] = ROOT

from desktop.splash import SplashScreen
import time

# 创建启动画面
splash = SplashScreen()

def on_ready():
    print("启动完成，关闭画面...")
    splash.close()
    print("✅ 启动画面测试通过")

# 自动播放动画，然后回调
splash.auto_progress(duration=2.0, callback=on_ready)
splash.root.mainloop()
print("✅ 启动画面已关闭")
