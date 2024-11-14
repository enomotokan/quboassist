use pyo3::prelude::*;
use pyo3::Python;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::cmp::min;
use std::sync::mpsc;
use std::thread;
use num_cpus;
use std::collections::BTreeMap as HashMap;
// use fxhash::FxHashMap;
// use std::collections::HashMap;

#[pyclass]

pub struct Lin {
    #[pyo3(get, set)]
    index_list: Vec<usize>,
    #[pyo3(get, set)]
    coef_list: Vec<f32>,
}

#[pymethods]

impl Lin {
    #[new]
    fn new(index_list: Vec<usize>, coef_list: Vec<f32>) -> Self {
        Lin{ index_list, coef_list }
    }

    fn append(&mut self, index: usize, a: f32) {
        self.index_list.push(index);
        self.coef_list.push(a);
    }
}

#[pyclass]
#[derive(Clone)]

pub struct Quad {
    #[pyo3(get, set)]
    index_list: Vec<usize>,
    #[pyo3(get, set)]
    index_list_list: Vec<Vec<usize>>,
    #[pyo3(get, set)]
    coef_list_list: Vec<Vec<f32>>,
}

#[pymethods]

impl Quad {
    #[new]
    fn new(index_list: Vec<usize>, index_list_list: Vec<Vec<usize>>, coef_list_list: Vec<Vec<f32>>) -> Self {
        Quad{ index_list, index_list_list, coef_list_list }
    }

    fn todense(&self) -> Vec<Vec<f32>>{

        // let dense = (0..self.index_list.len()).into_par_iter().map(|i| {

        //     let mut vec = vec![0.0; self.index_list.len()];
        //     let index0 = self.index_list[i];

        //     for j in 0..self.index_list_list[i].len() {
        //         let index1 = self.index_list_list[i][j];
        //         vec[index1] = self.coef_list_list[i][j];
        //     }

        //     return vec
        // }).collect();

        let mut dense = vec![vec![0.0; self.index_list.len()]; self.index_list.len()];

        for i in 0..self.index_list.len() {

            let index0 = self.index_list[i];

            for j in 0..self.index_list_list[i].len() {

                let index1 = self.index_list_list[i][j];
                dense[index0][index1] = self.coef_list_list[i][j];
            }
        }

        return dense
    }
}

#[pyfunction]

// calculate the squared formula

fn pow_lin_parallel (py: Python, lin: &Lin) -> PyResult<Py<Quad>> {

    let index_list: Arc<Vec<usize>> = Arc::new(lin.index_list.clone());
    let coef_list: Arc<Vec<f32>> = Arc::new(lin.coef_list.clone());
    let index_list_list: Arc<Vec<Mutex<Vec<usize>>>> = Arc::new((0..index_list.len()).map(|_| Mutex::new(vec![])).collect());
    let coef_list_list: Arc<Vec<Mutex<Vec<f32>>>> = Arc::new((0..index_list.len()).map(|_| Mutex::new(vec![])).collect());
    
    let num_thread: usize = min(index_list.len() / 2, num_cpus::get() / 8);
    let mut thread_hundles = vec![];

    if index_list.len() % 2 == 1 {
        let i = index_list.len() / 2;

        let mut index_list_list_i = index_list_list[i].lock().unwrap();
                let mut coef_list_list_i = coef_list_list[i].lock().unwrap();

                index_list_list_i.push(index_list[i]);
                coef_list_list_i.push(coef_list[i] * coef_list[i]);

                for j in 1..index_list.len() - i {
                    index_list_list_i.push(index_list[i + j]);
                    coef_list_list_i.push(2.0 * coef_list[i] * coef_list[i + j]);
                }

    }

    let mut work_lists: Vec<(usize, Arc<Vec<usize>>, Arc<Vec<f32>>, Arc<Vec<Mutex<Vec<usize>>>>, Arc<Vec<Mutex<Vec<f32>>>>)> = vec![];

    for n in 0..num_thread {
        work_lists.push((n, index_list.clone(), coef_list.clone(), index_list_list.clone(), coef_list_list.clone()));
    }

    for (n, index_list, coef_list, index_list_list, coef_list_list) in work_lists {

        thread_hundles.push(
            thread::spawn(move || {
                for k in 0..(index_list.len() / 2 - n - 1) / num_thread + 1  {

                    let i = num_thread * k + n;
            
                    let mut index_list_list_i = index_list_list[i].lock().unwrap();
                    let mut coef_list_list_i = coef_list_list[i].lock().unwrap();

                    index_list_list_i.push(index_list[i]);
                    coef_list_list_i.push(coef_list[i] * coef_list[i]);

                    for j in 1..&index_list.len() - i {
                        index_list_list_i.push(index_list[i + j]);
                        coef_list_list_i.push(2.0 * coef_list[i] * coef_list[i + j]);
                    }

                    let i = &index_list.len() - i - 1;
            
                    let mut index_list_list_i = index_list_list[i].lock().unwrap();
                    let mut coef_list_list_i = coef_list_list[i].lock().unwrap();

                    index_list_list_i.push(index_list[i]);
                    coef_list_list_i.push(coef_list[i] * coef_list[i]);

                    for j in 1..index_list.len() - i {
                        index_list_list_i.push(index_list[i + j]);
                        coef_list_list_i.push(2.0 * coef_list[i] * coef_list[i + j]);
                    }
                }
            })
        );
    }

    for handle in thread_hundles {
        handle.join().unwrap();
    }
    let index_list_list = index_list_list.par_iter().map(|vec| vec.lock().unwrap().clone()).collect();
    let coef_list_list = coef_list_list.par_iter().map(|vec| vec.lock().unwrap().clone()).collect();

    let return_coef = Py::new(py, Quad{ index_list: Arc::try_unwrap(index_list).unwrap(), index_list_list: index_list_list, coef_list_list: coef_list_list });

    return return_coef    
}

# [pyfunction]

fn pow_lin (py: Python, lin: &Lin) -> PyResult<Py<Quad>> {

    let index_list: &Vec<usize> = &lin.index_list;
    let coef_list: &Vec<f32> = &lin.coef_list;

    let mut index_list_list: Vec<Vec<usize>> = vec![];
    let mut coef_list_list: Vec<Vec<f32>> = vec![];

    for i in 0..index_list.len() {
        index_list_list.push(vec![]);
        coef_list_list.push(vec![]);

        index_list_list[i].push(index_list[i]);
        coef_list_list[i].push(coef_list[i] * coef_list[i]);

        for j in i + 1..index_list.len() {
            index_list_list[i].push(index_list[j]);
            coef_list_list[i].push(2.0 * coef_list[i] * coef_list[j]);
        }
    }

    let return_quad = Py::new(py, Quad {index_list: index_list.clone(), index_list_list: index_list_list, coef_list_list: coef_list_list }) ;

    return return_quad
}

# [pyfunction]

fn pow_cond_bin (py: Python, lin: &Lin, c: f32) -> PyResult<Py<Quad>> {

    let index_list: &Vec<usize> = &lin.index_list;
    let coef_list: &Vec<f32> = &lin.coef_list;

    let mut index_list_list: Vec<Vec<usize>> = vec![];
    let mut coef_list_list: Vec<Vec<f32>> = vec![];

    for i in 0..index_list.len() {
        index_list_list.push(vec![]);
        coef_list_list.push(vec![]);

        index_list_list[i].push(index_list[i]);
        coef_list_list[i].push((coef_list[i] + 2.0 * c) * coef_list[i]);

        for j in i + 1..index_list.len() {
            index_list_list[i].push(index_list[j]);
            coef_list_list[i].push(2.0 * coef_list[i] * coef_list[j]);
        }
    }

    let return_quad = Py::new(py, Quad {index_list: index_list.clone(), index_list_list: index_list_list, coef_list_list: coef_list_list }) ;

    return return_quad
}

# [pyfunction]

fn obj_bin (py: Python, quad: &Quad, lin: &Lin, index_binindptr: Vec<usize>, variable_A: Vec<Vec<f32>>, variable_range: Vec<Vec<f32>>) -> PyResult<Py<Quad>> {

    let mut return_quad: Quad = Quad{ index_list: vec![], index_list_list: vec![], coef_list_list: vec![] };

    let mut head_lin: usize = 0;
    let mut head_quad: usize = 0;
    let mut index_bin: usize;

    while head_quad < quad.index_list.len() {

        if lin.index_list[head_lin] > quad.index_list[head_quad] || head_lin == lin.index_list.len() {
        
            return_quad.index_list.extend(index_binindptr[quad.index_list[head_quad]]..index_binindptr[quad.index_list[head_quad]] + variable_A[quad.index_list[head_quad]].len());

            for head_quad_ in 0..quad.index_list_list[head_quad].len() {

                if quad.index_list[head_quad] == quad.index_list_list[head_quad][head_quad_] {
                    for k in 0..variable_A[quad.index_list[head_quad]].len() {

                        return_quad.index_list_list.push(vec![]);
                        return_quad.coef_list_list.push(vec![]);
                        index_bin = return_quad.index_list_list.len() - 1;

                        return_quad.index_list_list[index_bin].push(index_binindptr[quad.index_list[head_quad]] + k);
                        return_quad.coef_list_list[index_bin].push(quad.coef_list_list[head_quad][head_quad_] * variable_A[quad.index_list[head_quad]][k] * (variable_A[quad.index_list[head_quad]][k] + 2.0 * variable_range[quad.index_list[head_quad]][0]));

                        for k_ in k + 1..variable_A[quad.index_list[head_quad]].len() {
                            return_quad.index_list_list[index_bin].push(index_binindptr[quad.index_list[head_quad]] + k_);
                            return_quad.coef_list_list[index_bin].push(quad.coef_list_list[head_quad][head_quad_] * 2.0 * variable_A[quad.index_list[head_quad]][k] * variable_A[quad.index_list[head_quad]][k_]);
                        }
                    } 
                }
                else {
                    for k in 0..variable_A[quad.index_list[head_quad]].len() {

                        return_quad.index_list_list.push(vec![]);
                        return_quad.coef_list_list.push(vec![]);
                        index_bin = return_quad.index_list_list.len() - 1;

                        return_quad.index_list_list[index_bin].push(index_binindptr[quad.index_list[head_quad]] + k);
                        return_quad.coef_list_list[index_bin].push(quad.coef_list_list[head_quad][head_quad_] * variable_A[quad.index_list_list[head_quad][head_quad_]][k] * variable_A[quad.index_list_list[head_quad][head_quad_]][k]);

                        for k_ in 0..variable_A[quad.index_list[head_quad]].len() {
                            return_quad.index_list_list[index_bin].push(index_binindptr[quad.index_list_list[head_quad][head_quad_]]+ k_);
                            return_quad.coef_list_list[index_bin].push(quad.coef_list_list[head_quad][head_quad_] * 2.0 * variable_A[quad.index_list_list[head_quad][head_quad_]][k] * variable_A[quad.index_list_list[head_quad][head_quad_]][k_]);
                        }
                    } 
                }
            }
            
            head_quad += 1;
        }

        else if lin.index_list[head_lin] == quad.index_list[head_quad] {

            return_quad.index_list.extend(index_binindptr[quad.index_list[head_quad]]..index_binindptr[quad.index_list[head_quad]] + variable_A[quad.index_list[head_quad]].len());
            
            for head_quad_ in 0..quad.index_list_list[head_quad].len() {

                if quad.index_list[head_quad] == quad.index_list_list[head_quad][head_quad_] {
                    for k in 0..variable_A[quad.index_list[head_quad]].len() {

                        return_quad.index_list_list.push(vec![]);
                        return_quad.coef_list_list.push(vec![]);
                        index_bin = return_quad.index_list_list.len() - 1;

                        return_quad.index_list_list[index_bin].push(index_binindptr[quad.index_list[head_quad]] + k);
                        return_quad.coef_list_list[index_bin].push(lin.coef_list[head_quad] * variable_A[quad.index_list[head_quad]][k] + quad.coef_list_list[head_quad][head_quad_] * variable_A[quad.index_list[head_quad]][k] * variable_A[quad.index_list[head_quad]][k]);

                        for k_ in k + 1..variable_A[quad.index_list[head_quad]].len() {
                            return_quad.index_list_list[index_bin].push(index_binindptr[quad.index_list[head_quad]] + k_);
                            return_quad.coef_list_list[index_bin].push(quad.coef_list_list[head_quad][head_quad_] * 2.0 * variable_A[quad.index_list[head_quad]][k] * variable_A[quad.index_list[head_quad]][k_]);
                        }
                    } 
                }
                else {
                    for k in 0..variable_A[quad.index_list[head_quad]].len() {

                        return_quad.index_list_list.push(vec![]);
                        return_quad.coef_list_list.push(vec![]);
                        index_bin = return_quad.index_list_list.len() - 1;

                        return_quad.index_list_list[index_bin].push(index_binindptr[quad.index_list[head_quad]] + k);
                        return_quad.coef_list_list[index_bin].push(quad.coef_list_list[head_quad][head_quad_] * variable_A[quad.index_list_list[head_quad][head_quad_]][k] * variable_A[quad.index_list_list[head_quad][head_quad_]][k]);

                        for k_ in 0..variable_A[quad.index_list[head_quad]].len() {
                            
                            return_quad.index_list_list[index_bin].push(index_binindptr[quad.index_list_list[head_quad][head_quad_]]+ k_);
                            return_quad.coef_list_list[index_bin].push(quad.coef_list_list[head_quad][head_quad_] * 2.0 * variable_A[quad.index_list_list[head_quad][head_quad_]][k] * variable_A[quad.index_list_list[head_quad][head_quad_]][k_]);
                        }
                    } 
                }
            }

            head_lin += 1;
            head_quad += 1;

        }

        else {

            return_quad.index_list.extend(index_binindptr[lin.index_list[head_lin]]..index_binindptr[lin.index_list[head_lin]] + variable_A[lin.index_list[head_lin]].len());

            for k in 0..variable_A[lin.index_list[head_lin]].len() {
                return_quad.index_list_list.push(vec![index_binindptr[lin.index_list[head_lin]] + k]);
                return_quad.coef_list_list.push(vec![lin.coef_list[head_lin] * variable_A[lin.index_list[head_lin]][k]]);
            }

            head_lin += 1;
        }
    }

    while head_lin < lin.index_list.len() {
        return_quad.index_list.extend(index_binindptr[lin.index_list[head_lin]]..index_binindptr[lin.index_list[head_lin]] + variable_A[lin.index_list[head_lin]].len());

        for k in 0..variable_A[lin.index_list[head_lin]].len() {
            return_quad.index_list_list.push(vec![index_binindptr[lin.index_list[head_lin]] + k]);
            return_quad.coef_list_list.push(vec![lin.coef_list[head_lin] * variable_A[lin.index_list[head_lin]][k]]);
        }

        head_lin += 1;
    }

    return Py::new(py, return_quad)
}


# [pyfunction]

// search where the index is in the index list

fn where_list(index_list: Vec<usize>, index: usize) -> PyResult<(bool, usize)> {

    let mut left: usize = 0;
    let mut right: usize = index_list.len() - 1;
    let mut mid: usize;

    let result: (bool, usize);

    loop {
        mid = (left + right) / 2 as usize;
        
        if right == left + 1 {
            if index_list[left] == index {
                result = (true, left);
                break
            }
            
            else if index_list[right] == index {
                result = (true, right);
                break
            }

            else if index_list[right] < index {
                result = (false, right + 1);
                break
            }
            else {
                result = (false, right);
                break
            }
        }
        
        else {
            if index_list[mid] == index {
                result = (true, mid);
                break
            }

            else if index_list[mid] < index {
                left = mid
            }

            else {
                right = mid;
            }
        }
    }
    
    return Ok(result)
}

# [pyfunction]

fn prod_lin(py: Python, lin: &Lin, lin_: &Lin) -> PyResult<Py<Quad>> {

    let (sender_0, reciever_0) = mpsc::channel();
    let (sender_1, reciever_1) = mpsc::channel();

    thread::scope(|s| {

        s.spawn(move || {

            let mut head: usize = 0;
            let mut head_: usize = 0;

            let mut quad_0 = Quad { index_list: vec![], index_list_list: vec![], coef_list_list: vec![] };

            'outer: while head < lin.index_list.len() {

                while lin.index_list[head] > lin_.index_list[head_] {
                    head_ += 1;
                    if head_ == lin_.index_list.len() { break 'outer; }
                }

                quad_0.index_list.push(lin.index_list[head]);
                quad_0.index_list_list.push(lin_.index_list[head_..].to_vec());
                quad_0.coef_list_list.push(times_coef_list(lin.coef_list[head], lin_.coef_list[head_..].to_vec()).unwrap());

                head += 1;
            }

            sender_0.send(quad_0).unwrap();
        }).join().unwrap();

        s.spawn(move || {

            let mut head = 0;
            let mut head_ = 0;

            let mut quad_1 = Quad { index_list: vec![], index_list_list: vec![], coef_list_list: vec![] };

            'outer: while head_ < lin_.index_list.len() {

                while lin_.index_list[head_] >= lin.index_list[head] {
                    head += 1;
                    if head == lin.index_list.len() { break 'outer; }
                }

                quad_1.index_list.push(lin_.index_list[head_]);
                quad_1.index_list_list.push(lin.index_list[head..].to_vec());
                quad_1.coef_list_list.push(times_coef_list(lin_.coef_list[head_], lin.coef_list[head..].to_vec()).unwrap());

                head_ += 1;
            }

            sender_1.send(quad_1).unwrap();

        }).join().unwrap();
        
    });

    return Py::new(py, add_quad(py, &reciever_0.recv().unwrap(), &reciever_1.recv().unwrap(), 1.0, 1.0).unwrap())
}

# [pyfunction]

fn times_coef_list(a: f32, coef_list: Vec<f32>) -> PyResult<Vec<f32>> {

    let a_coef_list = coef_list.par_iter().map(|v| a * v).collect();

    return Ok(a_coef_list);
}

# [pyfunction]

fn times_lin(py: Python, a: f32, lin: &Lin) -> PyResult<Py<Lin>> {

    return Py::new(py, Lin{index_list: lin.index_list.clone(), coef_list: times_coef_list(a, lin.coef_list.clone()).unwrap() } );

}   

# [pyfunction]

fn times_coef_list_list(a: f32, coef_list_list: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
    let a_coef_list_list = coef_list_list.par_iter().map(|coef_list| coef_list.par_iter().map(|v| a * v).collect()).collect();
    return Ok(a_coef_list_list);
}

# [pyfunction]

fn times_quad(py: Python, a: f32, quad: &Quad) -> PyResult<Py<Quad>> {

    return Py::new(py, Quad{index_list: quad.index_list.clone(), index_list_list: quad.index_list_list.clone(), coef_list_list: times_coef_list_list(a, quad.coef_list_list.clone() ).unwrap() });
}

# [pyfunction]

fn add_lin(py: Python, lin: &Lin, lin_: &Lin, a: f32, a_: f32) -> PyResult<Py<Lin>> {

    if a == 0.0 && a_ == 0.0 {
        return Py::new(py, Lin { index_list: vec![], coef_list: vec![] })
    }
    else if a == 0.0 {
        return Py::new(py, times_lin(py, a_, lin_).unwrap())
    }
    else if a_ == 0.0 {
        return Py::new(py, times_lin(py, a, lin).unwrap())
    }

    let mut index_list = lin.index_list.clone();
    let mut coef_list = lin.coef_list.clone();

    let index_list_ = &lin_.index_list;
    let coef_list_ = &lin_.coef_list;
    
    let mut head: usize = 0;
    let mut head_: usize = 0;

    let mut s: f32;

    while head < index_list.len() && head_ < index_list_.len() {

        if index_list[head] == index_list_[head_] {
            coef_list[head] *= a;

            s = a * coef_list[head] + a_ * coef_list_[head_];
            
            if s != 0.0 {
                coef_list[head] = s;

                head += 1;
                head_ += 1;
            }
            else {
                index_list.remove(head);
                coef_list.remove(head);

                head_ += 1;
            }
        }

        else if index_list[head] < index_list_[head_] {
            coef_list[head] = a * coef_list[head];
            head += 1;
        }

        else {
            index_list.insert(head, index_list_[head_]);
            coef_list.insert(head, a_* coef_list_[head_]);

            head += 1;
            head_ += 1;
        }
    }

    if head < index_list.len() {
        for i in head..index_list.len() {
            coef_list[i] *= a;
        }
    }
    else if head_ < index_list_.len() {
        index_list.extend(&index_list_[head_..]);
        coef_list.extend(&coef_list_[head_..]);

        for i in head..index_list.len() {
            coef_list[i] *= a_;
        }
    }

    return Py::new(py, Lin { index_list: index_list, coef_list: coef_list })
}

# [pyfunction]

fn add_quad(py: Python, quad: &Quad, quad_: &Quad, a: f32, a_: f32) -> PyResult<Py<Quad>> {

    if (quad.index_list.len() == 0 && quad_.index_list.len() == 0) || (a == 0.0 && a_ == 0.0) {
        return Py::new(py, Quad {index_list: vec![], index_list_list: vec![], coef_list_list: vec![] })
    }
    else if quad.index_list.len() == 0 || a == 0.0 {
        return Py::new(py, times_quad(py, a_, quad_).unwrap())
    }
    else if  quad_.index_list.len() == 0 || a_ == 0.0 {
        return Py::new(py, times_quad(py, a, quad).unwrap())
    }

    let mut index_list = quad.index_list.clone();
    let mut index_list_list = quad.index_list_list.clone();
    let mut coef_list_list = quad.coef_list_list.clone();

    let index_list_ = &quad_.index_list;
    let index_list_list_ = &quad_.index_list_list;
    let coef_list_list_ = &quad_.coef_list_list;

    let mut head: usize = 0;
    let mut head_: usize = 0;

    let mut pattern: usize;

    while head < index_list.len() || head_ < index_list_.len() {

        if head < index_list.len() && head_ < index_list_.len() && index_list[head] == index_list_[head_] { pattern = 0; }
        else if head_ == index_list_.len() { pattern = 1; }
        else if head == index_list.len(){ pattern = 2;}
        else if index_list[head] < index_list_[head_] {pattern = 1;}
        else { pattern = 2;}

        match pattern {

            0 => {

                let lin_add = add_lin(py, &Lin { index_list: index_list_list[head].clone(), coef_list: coef_list_list[head].clone() }, &Lin { index_list: index_list_list_[head_].clone(), coef_list: coef_list_list_[head_].clone() }, a, a_).unwrap();

                index_list_list[head] = lin_add.borrow(py).index_list.clone();
                coef_list_list[head] = lin_add.borrow(py).coef_list.clone();

                head += 1;
                head_ += 1;
            }

            1 => {
                coef_list_list[head] = times_coef_list(a, coef_list_list[head].clone()).unwrap();
                head += 1;
            }

            _ => {
                index_list.insert(head, index_list_[head_]);
                index_list_list.insert(head, index_list_list_[head_].clone());
                coef_list_list.insert(head, times_coef_list(a_, coef_list_list_[head_].clone()).unwrap());

                head += 1;
                head_ += 1;
            }
        }
    }

    return Py::new(py, Quad { index_list: index_list, index_list_list: index_list_list, coef_list_list: coef_list_list})

}

# [pyfunction]

fn quad_todict(qubo: &Quad, binindex_index: Vec<Vec<usize>>) -> PyResult<HashMap<((usize, usize), (usize, usize)), f32>> {
    
    let indexes: Vec<((usize, usize), (usize, usize))> = (0..qubo.index_list.len()).into_par_iter().map(|i| (0..qubo.index_list_list[i].len()).map(|j| ((binindex_index[qubo.index_list[i]][0], binindex_index[qubo.index_list[i]][1]), (binindex_index[qubo.index_list[j]][0], binindex_index[qubo.index_list[j]][1]))).collect::<Vec<_>>()).collect::<Vec<_>>().concat();
    let values: Vec<f32> = (0..qubo.index_list.len()).into_par_iter().flat_map(|i| qubo.coef_list_list[i].clone()).collect::<Vec<_>>();

    let qubo_dwaveneal: HashMap<_, _> = indexes.into_iter().clone().zip(values.into_iter()).collect::<HashMap<_, _>>();

    Ok(qubo_dwaveneal)

}

/// A Python module implemented in Rust.
#[pymodule(name = "quboassistfunc")]

fn quboassistfunc(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_class::<Lin>()?;
    m.add_class::<Quad>()?;

    m.add_function(wrap_pyfunction!(pow_lin_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(pow_lin, m)?)?;
    m.add_function(wrap_pyfunction!(pow_cond_bin, m)?)?;
    m.add_function(wrap_pyfunction!(obj_bin, m)?)?;
    m.add_function(wrap_pyfunction!(where_list, m)?)?;
    m.add_function(wrap_pyfunction!(prod_lin, m)?)?;
    m.add_function(wrap_pyfunction!(times_coef_list, m)?)?;
    m.add_function(wrap_pyfunction!(times_lin, m)?)?;
    m.add_function(wrap_pyfunction!(times_coef_list_list, m)?)?;
    m.add_function(wrap_pyfunction!(times_quad, m)?)?;
    m.add_function(wrap_pyfunction!(add_lin, m)?)?;
    m.add_function(wrap_pyfunction!(add_quad, m)?)?;
    m.add_function(wrap_pyfunction!(quad_todict, m)?)?;

    Ok(())
}
