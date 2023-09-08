import streamlit as st
import joblib
from AutoML_module import *

if __name__ == "__main__":

    st.title('Экомониторинг v.1')
    st.text("""Экомониторинг: система контроля выбросов помощью цифрового двойникана основе машинного обучения""")
    # df_vertica_example = pd.DataFrame(
    #     columns=['tag_name', 'ts', 'num_val', 'str_val'],
    #     data=[['ZSN.05.01.03.Q23.00.0.30009_1101.L', '2023-01-02 18:00:00', '0.975', 'NaN']])
    # st.dataframe(df_vertica_example)

    upload_file1 = st.file_uploader("Выберете входные данные, для которых нужно провестти расчет (X_test_scaled)")
    if upload_file1 is not None:
        X_test_scaled = pd.read_csv(upload_file1, index_col='Время отбора', parse_dates=['Время отбора'])
        st.dataframe(X_test_scaled)
        # st.write(df_vertica.describe(include='all'))
    else: 
        st.warning("Нужно загрузить данные для анализа (X_test_scaled)")

    upload_file2 = st.file_uploader("Выберете целевые значения, для проверки модели на тестировочной выборке (y_test)")
    if upload_file2 is not None:
        y_test = pd.read_csv(upload_file2, index_col='Время отбора', parse_dates=['Время отбора'])
        st.dataframe(y_test)
        # st.write(df_vertica.describe(include='all'))
    else: 
        st.warning("Нужно загрузить данные для анализа (y_test)")

    upload_file3 = st.file_uploader("Выберете входные данные, на которых обучить модель (X_train_scaled)")
    if upload_file3 is not None:
        X_train_scaled = pd.read_csv(upload_file3, index_col='Время отбора', parse_dates=['Время отбора'])
        #st.dataframe(X_train_scaled)
        # st.write(df_vertica.describe(include='all'))
    else: 
        st.warning("Нужно загрузить данные для анализа (X_train_scaled)")    

    upload_file4 = st.file_uploader("Выберете целевые значения, для обучения модели на обучающей выборке (y_train)")
    if upload_file4 is not None:
        y_train = pd.read_csv(upload_file4, index_col='Время отбора', parse_dates=['Время отбора'])
        #st.dataframe(y_train)
        # st.write(df_vertica.describe(include='all'))
    else: 
        st.warning("Нужно загрузить данные для анализа (y_train)")

    upload_file5 = st.file_uploader("Загрузите модель (optimized_regressor.pkl)")
    if upload_file5 is not None:
        model = joblib.load(upload_file5)
    else: 
        st.warning("Нужно загрузить мдель (optimized_regressor.pkl)")

    target = st.text_input('Введите название таргета, например: "target"')
    y_test_pred = None
    if st.button("Провести расчет для тестовой выборки"):
        if upload_file1 is None:
            st.warning("Нужно загрузить все данные (X_test_scaled)")
        elif upload_file2 is None:
            st.warning("Нужно загрузить все данные (y_test)")
        elif upload_file2 is None:
            st.warning("Нужно загрузить все данные (X_train_scaled)")
        elif upload_file2 is None:
            st.warning("Нужно загрузить все данные (y_train)")
        else: 
            st.text("Расчет начат...")
            AutoML_model = AutoML(model, X_test_scaled, target, y_test, 
                           X_train_scaled, y_train)
            y_test_pred, y_train_pred = AutoML_model.make_inference()
            st.text("Расчет по тестовым данным завершен...")
            st.text("Построение графика...")
            fig = AutoML_model.plot_results()
            st.pyplot(fig)

            metrics = AutoML_model.calculate_metrics()
            st.dataframe(metrics)


    # if y_test_pred is not None:
    #     if st.button("Вывести метрики расчета"):
    #         # fig = AutoML_model.plot_results()
    #         # st.pyplot(fig)
    #         metrics = AutoML_model.calculate_metrics()
    #         st.dataframe(metrics)

    if y_test_pred is not None:
        st.download_button(
            label="Сохранить результаты",
            data=y_test_pred.to_csv(), 
            file_name= f'Предсказанные значения выбросов {(target)}.csv')
