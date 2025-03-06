import config
import text_processing_utils
import Calculate_LSM
import Calculate_rLSM
import Calculate_FW


def main():
    # 获取功能词典
    function_words = text_processing_utils.load_function_words(config.FUNCTION_WORDS_FILE)

    Calculate_FW.process_all_files(config.ROOT_DIRECTORY, config.OUTPUT_DIRECTORY, function_words)

    # 计算LSM值
    Calculate_rLSM.process_multiple_xlsx(config.OUTPUT_DIRECTORY, config.SPLIT_DIRECTORY, config.FUNCTION_WORDS_FILE)


if __name__ == "__main__":
    main()
