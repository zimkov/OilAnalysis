import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from pathlib import Path
import os
from scipy.optimize import curve_fit

from calcutalor import Calculator

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


JSON_FILE_PATH = Path("function-definitions.json")
@app.get("/get-json", response_class=JSONResponse)
async def get_json():
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": "JSON file not found"}
        )
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500,
            content={"error": "Error decoding JSON file"}
        )

@app.post("/calculate/")
async def calculate(startValues=Form(...), minValues=Form(...), maxValues=Form(...), coefs=Form(...), qcoefs=Form(...), normValues=Form(...)):
    startValues_vals = json.loads(startValues)
    minValues_vals = json.loads(minValues)
    maxValues_vals = json.loads(maxValues)
    coefs_vals = json.loads(coefs)
    qcoefs_vals = json.loads(qcoefs)
    normValues_vals = json.loads(normValues)

    calc = Calculator(startValues_vals, minValues_vals, maxValues_vals, coefs_vals, qcoefs_vals, normValues_vals)

    # 3. Первоначальный расчет (без оптимизации)
    time_intervals = np.linspace(0, 1, 11)
    initial_solution = calc.calculate(time_intervals)
    
    # 4. Оптимизация коэффициентов!
    target = np.clip(initial_solution, minValues_vals, maxValues_vals)  # Целевые значения
    optimization_result = calc.adjust_coefficients(target)  # Вызов оптимизации
    
    # 5. Финальный расчет с оптимизированными коэффициентами
    final_solution = calc.calculate(time_intervals)
    # final_solution = initial_solution

# Список всех параметров с названиями
    params = [
        ('L1', 'Время испарения'),
        ('L2', 'Время ликвидации'),
        ('L3', 'Площадь заражения'),
        ('L4', 'Время подхода облака'),
        ('L5', 'Потери первичного облака'),
        ('L6', 'Потери вторичного облака'),
        ('L7', 'Получившие амбулаторную помощь'),
        ('L8', 'Размещенные в стационаре'),
        ('L9', 'Количество поражённой техники'),
        ('L10', 'Растворы обеззараживания местности'),
        ('L11', 'Силы и средства для спас. работ'),
        ('L12', 'Эфф. системы оповещения'),
        ('L13', 'Людей в зоне поражения'),
        ('L14', 'Спасателей в зоне поражения'),
        ('L15', 'Развитость системы МЧС')
    ]

    # Создаем группы по 3 графика
    group_size = 3
    num_groups = (len(params) + group_size - 1) // group_size
    plot_images = []

    for group_idx in range(num_groups):
        # Создаем новую фигуру для группы
        fig, axs = plt.subplots(group_size, 1, figsize=(16, 8 * group_size))
        plt.subplots_adjust(hspace=0.5)
        
        # Получаем подмножество параметров для текущей группы
        start_idx = group_idx * group_size
        end_idx = min((group_idx + 1) * group_size, len(params))
        group_params = params[start_idx:end_idx]
        
        # Строим графики для каждой группы
        for idx, (param, ax) in enumerate(zip(group_params, axs.flatten())):
            # Получаем данные для текущего параметра
            param_idx = start_idx + idx
            y_values = final_solution[:, param_idx]
            
            # Строим график
            line, = ax.plot(time_intervals, y_values, 
                        label=f'{param[0]}: {param[1]}',
                        marker='o', markersize=5,
                        linewidth=2, alpha=0.8)
            
            # Настройки оформления
            ax.set_title(f'{param[0]}: {param[1]}', fontsize=12, pad=15)
            ax.set_xlabel('Время', fontsize=10)
            ax.set_ylabel('Значение', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right', fontsize=10)
            
            # Форматирование осей
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_ylim(calc.minValues[param_idx] * 0.9, calc.maxValues[param_idx] * 1.1)
            
            # Добавляем горизонтальные линии для min/max
            ax.axhline(y=calc.minValues[param_idx], color='r', linestyle='--', linewidth=1)
            ax.axhline(y=calc.maxValues[param_idx], color='g', linestyle='--', linewidth=1)
        
        # Удаляем пустые subplots
        for ax in axs.flatten()[len(group_params):]:
            fig.delaxes(ax)
        
        # Сохранение в буфер
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_images.append(base64.b64encode(buf.read()).decode('utf-8'))
        plt.close(fig)

# Теперь plot_images содержит список base64-строк для каждой группы графиков

    # Названия категорий
    categories = [
        "Время испарения", "Время ликвидации", "Площадь заражения", "Время подхода облака", "Потери первичного облака",
        "Потери вторичного облака", "Получившие амбулаторную помощь", "Размещенные в стационаре", "Количество поражённой техники",
        "Растворы обеззараживания местности", "Силы и средства для спас. работ", "Эфф. системы оповещения", "Людей в зоне поражения",
        "Спасателей в зоне поражения", "Развитость системы МЧС"
    ]

    # Углы для категорий
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # Допустим, maxValues — это список из максимальных значений для каждой категории
    maxValues = calc.maxValues.tolist()
    minValues = calc.minValues.tolist()

    maxValues += [maxValues[0]]
    minValues += [minValues[0]]

    assert len(angles) == len(maxValues), f"Размеры angles ({len(angles)}) и maxValues ({len(maxValues)}) должны совпадать!"

    # Строим графики для каждого времени
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))
    

    for i, ax in enumerate(axes.flat):
        # Вычисляем пропорциональный индекс для текущего i
        idx = int(i * (len(final_solution) - 1) / 5)  # Пропорционально делим массив solution

        # Получаем значения для момента времени idx
        values = final_solution[idx].tolist()

        # Замыкаем полигон (чтобы соединить последний лепесток с первым)
        values += values[:1]

        # Строим основную диаграмму
        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.plot(angles, values, color='blue', linewidth=2)

        # Добавляем линию для maxValues
        ax.plot(angles, maxValues, color='red', linewidth=2, linestyle='--', label='Max Values')
        ax.plot(angles, minValues, color='green', linewidth=1, linestyle='--', label='Min Values')

        # Заливка допустимой области
        ax.fill_between(
            angles, 
            minValues,
            maxValues,
            color='yellow',
            alpha=0.1,
            label='Допустимая область'
        )

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)

        # Вычисляем пропорциональный индекс для time_intervals
        time_idx = int(i * (len(time_intervals) - 1) / 5)
        ax.set_title(f't = {round(time_intervals[time_idx], 2)}', size=16, y=1.1)

    # Добавляем легенду
    plt.legend(loc='upper right')

    # Устанавливаем плотную компоновку
    plt.tight_layout()

    # Сохранение второго графика в буфер
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png")
    buf2.seek(0)
    img_str2 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    # Закрытие фигур после сохранения
    plt.close(fig2)

    # Аппроксимация для каждого параметра
    equations = []
    time_points = time_intervals
    
    for param_idx in range(final_solution.shape[1]):
        y_data = final_solution[:, param_idx]
        
        # 1. Выбираем тип аппроксимации (кубический полином)
        def poly_func(x, a, b, c, d):
            return a*x**3 + b*x**2 + c*x + d
        
        # 2. Подбор коэффициентов
        try:
            coeffs, _ = curve_fit(poly_func, time_points, y_data)
            a, b, c, d = coeffs
        except:
            a = b = c = d = 0
        
        # 3. Форматирование уравнения
        equation = (
            f"L{param_idx+1}(t) = "
            f"{round(a, 4)}t³ + {round(b, 4)}t² + {round(c, 4)}t + {round(d, 4)}"
        )
        equations.append(equation)

    return {
        "image1": plot_images,
        "image2": img_str2,
        "equations": equations
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)