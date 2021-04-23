class CodeWrapper:
    @staticmethod
    def wrap_with_function(src, wrapper_function_name) -> str:
        """
        wraps code, containing 'return smth' as a last line with function
        @param src: code to wrap
        @param wrapper_function_name: name of a function to wrap code with
        """
        res = ''
        res += f'def {wrapper_function_name}():\n'

        tabbed_src = '\n'.join([('\t' + line) for line in src.split('\n')])
        res += tabbed_src
        return res
