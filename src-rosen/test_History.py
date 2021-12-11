from Rosenbrock import History

history=History(10)
v=5.0
history.add_v(v)

assert history.get()==v, 'must equal {:}'.format(v)

history.add_v(0)
nhistory=history.n
v2=v*(nhistory-1)/nhistory
assert history.get()==v2, 'must equal {:}'.format(v2)

v3=-10.0
history.add_v(v3)
w3=(v*(nhistory-2)+v3)/nhistory
print(history.get())
assert history.get()==w3, 'must equal {:}'.format(w3)

print('test_passed')