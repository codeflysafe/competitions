import logging
import os.path
import time

def init_log(model_name,save_log = False):
    """
    :param model_name:
    :return:
    """
    # 记录器对象。注意 永远 不要直接实例化记录器，应当通过模块级别的函数 logging.getLogger(name) 。
    # 多次使用相同的名字调用 getLogger() 会一直返回相同的 Logger 对象的引用。
    logger = logging.getLogger()
    # setLevel(level)：给记录器设置阈值为 level 。
    # 日志等级小于 level 会被忽略。严重性为 level 或更高的日志消息将由该记录器的任何一个或多个处理器发出，除非将处理器的级别设置为比 level 更高的级别。
    logger.setLevel(logging.INFO)
    # Formater对象用于配置日志信息的最终顺序、结构和内容。与logging.Handler基类不同的是，应用代码可以直接实例化Formatter类。
    formatter = logging.Formatter(f"[{model_name}]->" + "%(asctime)s - %(filename)s[line:%(lineno)d] -%(levelname)s:%(message)s" )
    if save_log:
        rq = time.strftime('%Y%m%d', time.localtime(time.time()))
        pid = os.getpid()
        curPath = os.path.dirname(os.path.realpath(__file__))
        log_path = os.path.join(curPath,f'log/{rq}')
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        log_name = os.path.join(log_path, f'{pid}_{model_name}_{rq}.log')
        # 将日志消息发送到磁盘文件，默认情况下文件大小会无限增长
        fh = logging.FileHandler(log_name, mode='w',encoding='utf-8')
        # setLevel(level)：给处理器设置阈值为 level。
        # 日志级别小于 level 将被忽略。创建处理器时，日志级别被设置为 NOTSET （所有的消息都会被处理）
        # debug<info<warning<error<critical
        fh.setLevel(logging.DEBUG)
        # setFormatter(fmt): 将此处理器的 Formatter 设置为 fmt。
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # tell the handler to use this format
    console.setFormatter(formatter)
    # addHandler(hdlr):将指定的处理器hdlr添加到此记录器。

    logger.addHandler(console)
    return logger





