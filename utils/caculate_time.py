import cProfile
import pstats
import io
import time
# from examples.nature_cross import mask as mtd_main

from examples.nature_cross import per_no_mask1111 as mtd_main
# 定义一些示例函数
def child_function_1():
    time.sleep(1)  # 模拟耗时操作


def child_function_2():
    time.sleep(2)  # 模拟更长的耗时操作
    child_function_1()  # 调用子函数


def child_function_3():
    time.sleep(0.5)  # 模拟较短的耗时操作
    child_function_2()  # 调用子函数


def main_function():
    print("Starting main_function")
    child_function_1()
    child_function_2()
    child_function_3()
    print("Finished main_function")


# 使用 cProfile 运行主函数并获取性能分析数据
def profile_main_function():
    pr = cProfile.Profile()
    pr.enable()

    # 执行主函数
    # main_function()
    mtd_main()
    pr.disable()

    # 将性能分析结果输出到字符串
    s = io.StringIO()
    sortby = 'cumulative'  # 根据累积时间排序
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    # 将结果保存到文本文件
    name = "mtd1"
    with open(name + '_performance_analysis_1.txt', 'w') as f:
        f.write(s.getvalue())


if __name__ == "__main__":
    profile_main_function()