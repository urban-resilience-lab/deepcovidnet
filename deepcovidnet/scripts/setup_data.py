import os

data_dir = '/data/'

def get_weekly_data():
        weekly_data_dir = os.path.join(data_dir, 'weekly_patterns')
        
        if not os.path.isdir(weekly_data_dir):
                os.system('mkdir ' + weekly_data_dir)

        os.system('aws s3 sync s3://sg-c19-response/weekly-patterns/v1/main-file/ ' + weekly_data_dir)
        
        for f in os.listdir(weekly_data_dir):
                if f.endswith('.gz'):
                        f = os.path.join(weekly_data_dir, f)
                        os.system('gunzip ' + f)

def get_monthly_data():
        files = [
                'Jan20-AllPatterns-PATTERNS-2020_01-2020-03-23.zip',
                'Feb20-AllPatterns-PATTERNS-2020_02-2020-03-23.zip',
                'March2020-PATTERNS-2020_03-2020.zip'
        ]

        saved_file_names = [
                '2001-AllPatterns-PATTERNS-2020_01',
                '2002-AllPatterns-PATTERNS-2020_02',
                '2003-AllPatterns-PATTERNS-2020_03'
        ]

        monthly_dir = os.path.join(data_dir, 'monthly_patterns')
        
        for i in range(len(saved_file_names)):
                saved_file_names[i] = os.path.join(monthly_dir, saved_file_names[i])

        if not os.path.isdir(monthly_dir):
                os.system('mkdir ' + monthly_dir)
        
        for i, f in enumerate(files):
                if not os.path.exists(os.path.join(monthly_dir, f)):
                        os.system('aws s3 cp s3://sg-c19-response/historicpatterns/' + f + ' ' + monthly_dir)

                os.system('unzip ' + os.path.join(monthly_dir, f) + ' -d ' + saved_file_names[i])
                for sub_f in os.listdir(saved_file_names[i]):
                        if sub_f.endswith('.gz'):
                                sub_f = os.path.join(saved_file_names[i], sub_f)
                                os.system('gunzip ' + sub_f)

def get_social_distancing_data():
        sg_dir = os.path.join(data_dir, 'social_distancing')
        
        if not os.path.isdir(sg_dir):
            os.system('mkdir ' + sg_dir)
            os.system('aws s3 sync s3://sg-c19-response/social-distancing/v1/ ' + sg_dir)

        for d1 in os.listdir(os.path.join(sg_dir, '2020')):
            for d2 in os.listdir(os.path.join(sg_dir, '2020', d1)):
                for zipfile in os.listdir(os.path.join(sg_dir, '2020', d1, d2)):
                    os.system('gunzip ' + os.path.join(sg_dir, '2020', d1, d2, zipfile))

#get_weekly_data()
#get_monthly_data()
get_social_distancing_data()
