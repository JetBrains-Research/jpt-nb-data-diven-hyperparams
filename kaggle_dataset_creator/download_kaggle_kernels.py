from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.rest import ApiException
from tqdm import tqdm


class KernelDownloader:
    def __init__(self, model_name, model_name_patterns):
        self.api = KaggleApi()
        self.api.authenticate()

        self.model_name = model_name
        self.model_name_patterns = model_name_patterns

    def get_all_kernels_list(self):
        all_kernels_list = []
        for pattern in self.model_name_patterns:
            all_kernels_list += self.__get_all_kernels(pattern)
        all_kernels_list = KernelDownloader.__filter_unique_kernels(all_kernels_list)
        return all_kernels_list

    def get_all_kernel_refs_list(self):
        all_kernels_list = self.get_all_kernels_list()
        return [kernel.ref for kernel in all_kernels_list]

    def download_all_kernels(self, directory):
        kernel_refs = self.get_all_kernel_refs_list()
        for kernel_ref in tqdm(kernel_refs):
            try:
                self.api.kernels_pull(kernel_ref, directory)
            except ApiException as ae:
                print(f'{ae.__class__.__name__}: {str(ae)}')
                print(f'kernel_ref={kernel_ref}')

    def __get_all_kernels(self, search):
        result = []
        curr_page_no = 1

        while True:
            kernels = self.api.kernels_list(search=search, page_size=100, page=curr_page_no)
            if len(kernels) == 0:
                break
            result += kernels
            curr_page_no += 1
        return result

    @staticmethod
    def __kernels_contain(kernels, kernel):
        for k in kernels:
            if k.__dict__ == kernel.__dict__:
                return True
        return False

    @staticmethod
    def __filter_unique_kernels(kernels):
        unique_kernels = []

        for i in range(len(kernels)):
            if not KernelDownloader.__kernels_contain(unique_kernels, kernels[i]):
                unique_kernels.append(kernels[i])
        return unique_kernels


def main():
    rfc_patterns = ['random forest classifier', 'rfc',
                    'RandomForestClassifier', 'rf classifier',
                    'random forest'
                    ]
    rfc_kern_downloader = KernelDownloader('RFC', model_name_patterns=rfc_patterns)
    rfc_kern_downloader.download_all_kernels(directory='../../kaggle_data')


if __name__ == '__main__':
    main()
