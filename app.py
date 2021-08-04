from flask import Flask, render_template , request
import joblib

# instance of an app
app = Flask(__name__)

model = joblib.load('dib_lr.pkl')


@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/blogs' , methods= ['POST'])
def contact():
    a = request.form.get('area')
    b = request.form.get('rooms')
    c = request.form.get('bathroom')
    d = request.form.get('floors')
    e = request.form.get('driveway')
    f = request.form.get('game_room')
    g = request.form.get('cellar')
    h = request.form.get('gas')
    i = request.form.get('air')
    j = request.form.get('garage')
    k = request.form.get('situation')


    pred = model.predict([[int(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h),int(i),int(j),int(k)]])

    
    return render_template('blogs.html' , predicted_text = f'the house price is {pred}')

# run the app
if __name__ == '__main__':
    app.run(debug=True)